"""Scoring logic for the model router: combine base + learned scores."""
from __future__ import annotations

from typing import Any, Mapping, Optional

# Sharper grade scale: A is clearly better than C (1.0 vs 0.7 vs 0.5).
# This matters because a 1.2B lab model at 160 t/s (grade C) should NOT
# beat a 30B coder at 49 t/s (grade A) — the user's JSON says so.
GRADE_WEIGHTS = {
    "A": 1.0,
    "B": 0.8,
    "C": 0.6,
    "D": 0.4,
    "E": 0.25,
    "F": 0.15,
}
# Default (no grading) leans slightly toward "unknown" — treat as 0.5 so
# we don't silently demote ungraded endpoints to the bottom.
_GRADE_NONE = 0.5

# Learning rate for EMA updates. Higher = faster to react to new data.
EMA_ALPHA = 0.1


def grade_weight(grade: Optional[str]) -> float:
    g = (grade or "").upper()
    if g == "":
        return _GRADE_NONE
    return GRADE_WEIGHTS.get(g, _GRADE_NONE)


def speed_weight(tps: Optional[float], tps_ref: float = 100.0) -> float:
    """Map a tokens/s value to a 0..1 weight. Saturates at tps_ref."""
    if not tps or tps <= 0:
        return 0.1
    return min(1.0, (tps or 0) / tps_ref)


def _size_bonus(param_size: Optional[str], grade: Optional[str]) -> float:
    """Optional bonus for A/B models with a real parameter size (parsed from
    the template). Zero when the template doesn't record it, or the grade
    is below B. This is a tiebreaker, not the main driver — see base_score.
    """
    if not param_size:
        return 0.0
    g = (grade or "").upper()
    if g not in ("A", "B"):
        return 0.0
    text = str(param_size).strip().lower()
    for unit in ("b", "m"):
        if unit in text:
            try:
                n = float(text.split(unit)[0])
            except ValueError:
                return 0.0
            if unit == "b":
                return min(1.0, n / 30.0)  # saturates at 30B
            if unit == "m":
                return min(1.0, n / 1000.0)
            break
    return 0.0


def base_score(base: Mapping[str, Any]) -> float:
    """Compute an endpoint's base score from the template record.

    0.5 * grade + 0.3 * speed + 0.2 * tool_support
    Plus a 0.1 bonus if the model has a real parameter size AND an A/B
    grade (tiebreaker — does not dominate grade or speed).
    """
    g = grade_weight(base.get("grade"))
    s = speed_weight(base.get("tps"))
    t = 1.0 if base.get("supports_tools") else 0.0
    sz = _size_bonus(base.get("param_size"), base.get("grade"))
    if sz > 0:
        # Bonus path: redistribute from speed.
        return 0.5 * g + 0.2 * s + 0.2 * t + 0.1 * sz
    return 0.5 * g + 0.3 * s + 0.2 * t


def learned_score(learned: Mapping[str, Any]) -> Optional[float]:
    """Compute an endpoint's empirical score from the learned state.

    Returns None (no data yet) — caller should then apply only base_score.
    Otherwise: 0.6 * success_rate + 0.3 * speed_weight(learned_tps) + 0.1 * (1 - latency_norm)
    """
    if not learned:
        return None
    # Treat a zero-sample record as "no data" so a freshly-bootstrapped
    # endpoint doesn't pick up a neutral 0.5 that dilutes its base score.
    if not learned.get("sample_count"):
        return None
    success = float(learned.get("success_rate", 0.5))
    tps = float(learned.get("tokens_s") or 0)
    latency = float(learned.get("avg_latency_ms") or 0)
    latency_weight = 1.0 if latency <= 0 else max(0.0, 1.0 - min(1.0, latency / 500.0))
    return 0.6 * success + 0.3 * speed_weight(tps) + 0.1 * latency_weight


def task_matches(task: str, base: Mapping[str, Any]) -> bool:
    """Whether the endpoint's base record supports this task."""
    known = ("code", "tool_calling", "code_generation", "title",
             "compression", "vision", "approval")
    if task not in known:
        return True
    caps = base.get("task_capabilities") or []
    return task in caps


def final_score(
    base: Mapping[str, Any],
    learned: Optional[Mapping[str, Any]],
    task_learned: Optional[Mapping[str, Any]],
    task: str,
) -> float:
    """Combine base + learned scores for a given task.

    0.4 base + 0.3 overall-learned + 0.3 task-learned  (when we have both, non-zero sample counts)
    0.5 base + 0.5 overall-learned                     (when only overall, non-zero sample counts)
    base - truncation_penalty                           (cold start — no sample data yet)

    `truncation_ema` is tracked in `learned` and penalized to deprioritize
    endpoints that frequently truncate their streams.
    """
    b = base_score(base)
    l = learned_score(learned) if learned else None
    tl = learned_score(task_learned) if task_learned else None

    # Extract truncation data
    trunc_ema = max(0.0, min(1.0, (learned or {}).get("truncation_ema", 0.0)))
    sample_count = int((learned or {}).get("sample_count", 0))
    trunc_pen = truncation_penalty(trunc_ema, sample_count)

    if l is not None and tl is not None:
        return 0.4 * b + 0.3 * l + 0.3 * tl - trunc_pen
    if l is not None:
        return 0.5 * b + 0.5 * l - trunc_pen
    return b - trunc_pen


def update_ema(old: float, new: float) -> float:
    """Standard EMA update with the module's alpha."""
    return (1 - EMA_ALPHA) * old + EMA_ALPHA * new


def truncation_penalty(truncation_ema: float, sample_count: int) -> float:
    """Penalize endpoints that frequently truncate their streams.

    0.0 → 1.0 (truncation rate)
    - 10% truncation: 0 penalty
    - 75% truncation: 0.4 penalty (25% of total score)

    Applied to final_score to reduce routing to unstable streamers.
    """
    if sample_count < 3:
        return 0.0  # not enough data
    if truncation_ema <= 0.1:
        return 0.0
    if truncation_ema >= 0.75:
        return 0.4
    # Linear scaling between 0.1 and 0.75
    return min(0.4, max(0.0, ((truncation_ema - 0.1) / 0.65) * 0.4))

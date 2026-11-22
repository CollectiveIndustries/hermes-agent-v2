"""Model Router delegation backend — task-centric, learns endpoint scores.

Routes every `delegate_task` to the best-scoring endpoint for the task,
balancing the template's base grade/speed against empirical success,
latency, and tokens/s learned by observing live traffic on this host.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Mapping, Optional

from agent.delegation_provider import (
    DelegationProvider, DelegationRouteError,
)

from . import _scoring as sc
from . import _state as st

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Task classification
# ──────────────────────────────────────────────────────────────────────────────

# Keywords for canonical task ids. Order in the dict matters: earlier tasks
# are checked first when scoring is tied.
_TASK_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tool_calling":  ("tool", "api call", "http", "endpoint", "fetch",
                      "scrape", "web search", "browser", "weather api",
                      "send request", "call the"),
    "code_generation": ("boilerplate", "scaffold", "starter template", "generate a"),
    "vision":        ("image", "screenshot", "ocr", "visual", "diagram"),
    "approval":      ("approval", "approve", "classify", "gate", "judge"),
    "compression":   ("compress", "condense", "shorten"),
    "title":         ("headline", "summary title", "session title"),
    "code":          ("code", "coding", "implement", "function", "class", "bug",
                      "refactor", "unit test", "pytest", "script", "python",
                      "javascript", "rust", "go", "compile", "lint", "fastapi",
                      "module", "library"),
}
# If no keyword matches, pick based on simple heuristics.
_DEFAULT_TASK = "code"


def _classify_task(goal: str) -> str:
    """Infer a canonical task id from the goal text using a simple keyword score.

    The task with the most keyword hits wins. Ties resolve to the task listed
    first in this file's `_TASK_KEYWORDS` (tool_calling first, then ..., then code).
    """
    g = (goal or "").lower()
    if not g.strip():
        return _DEFAULT_TASK

    best_task: str = _DEFAULT_TASK
    best_hits: int = 0
    for task, kws in _TASK_KEYWORDS.items():
        hits = 0
        for kw in kws:
            if kw in g:
                hits += 1
                if hits > best_hits:
                    best_hits = hits
                    best_task = task
                break  # one hit per task is enough for this ordering
    return best_task


# ──────────────────────────────────────────────────────────────────────────────
# Provider class
# ──────────────────────────────────────────────────────────────────────────────

class ModelRouterProvider(DelegationProvider):
    """Route each `delegate_task` to the best endpoint for the inferred task."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: dict[str, Any] = {}
        self._template: dict[str, Any] = {}
        self._dirty = False
        # Throttled save: coalesce writes into at most one per _SAVE_PERIOD seconds.
        self._last_save = 0.0
        self._SAVE_PERIOD = 10.0

    # ── identity ──────────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        return "model-router"

    @property
    def display_name(self) -> str:
        return "Model Router (learned)"

    @property
    def summary(self) -> str:
        return ("Routes each delegate_task to the best endpoint for its inferred task "
                "(code / tool_calling / vision / ...), balancing template grade + speed "
                "against empirical success, latency, and tokens/s learned from live traffic.")

    @property
    def route_keys(self) -> tuple[str, ...]:
        return ()  # no extra per-route settings

    def is_available(self) -> bool:
        return True

    # ── bootstrap ─────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._state:
                return
            self._template = st.load_template()
            self._state = st.load_state()
            # Seed the endpoint list from the template. We deliberately
            # DO NOT seed `learned` or `task_scores` here — they fill in
            # as we observe live traffic (see _observe). A freshly
            # bootstrapped endpoint therefore scores purely on base_grade
            # + base_speed (no neutral 0.5 that would dilute it).
            for key, base in self._template.get("endpoints", {}).items():
                ep = self._state["endpoints"].setdefault(key, {
                    "base": dict(base),
                    "learned": None,   # filled in by _observe()
                    "task_scores": {},
                })
                # Template is the source of truth for base facts.
                ep["base"] = dict(base)
            # Endpoints we saw in prior runs but that are NOT in the current
            # template — keep them with whatever base we have (they may have
            # been added by the user, or removed from the catalog).

    # ── routing ───────────────────────────────────────────────────────────────
    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        goal = str(task.get("goal") or task.get("context") or "")
        task_id = _classify_task(goal)

        # Build the candidate list: every endpoint that claims to support this task.
        candidates: list[tuple[str, float]] = []
        with self._lock:
            for key, ep in self._state["endpoints"].items():
                base = ep.get("base") or {}
                if not sc.task_matches(task_id, base):
                    continue
                b = base
                l = ep.get("learned") or None
                tl = (ep.get("task_scores") or {}).get(task_id) or None
                score = sc.final_score(b, l, tl, task_id)
                candidates.append((key, score))

        if not candidates:
            # No endpoint in the catalog supports the inferred task. Fall back to
            # whatever the default delegation section already says (empty = inherit
            # parent's route). Refuse only if there is no catalog at all.
            if not self._template.get("endpoints"):
                raise DelegationRouteError(
                    f"No model_routing template found. "
                    f"Put a llama_out.json (or equivalent) in $HERMES_HOME "
                    f"or ~, or set `delegation.backend: default`."
                )
            logger.debug("no endpoint matched task %r; inheriting parent route", task_id)
            return {k: v for k, v in config.items() if k not in ("service_tier", "backends")}

        # Pick the best-scoring endpoint, break ties by the most samples.
        candidates.sort(key=lambda kv: (-kv[1],))
        best_key, best_score = candidates[0]
        with self._lock:
            ep = self._state["endpoints"][best_key]
            base = ep["base"]
            host, port, model = base["host"], base["port"], base["model"]
            base_url = f"http://{host}:{port}/v1"

        return {
            "provider": "custom",
            "model": model,
            "base_url": base_url,
            "api_key": "ollama",  # placeholder for local Ollama endpoints
            "label": f"{model[:32]}… (score {best_score:.2f})",
        }

    # ── learning hooks ────────────────────────────────────────────────────────
    def _observe(
        self, endpoint_id: Optional[str], task_id: str, *,
        ok: bool, latency_ms: float = 0.0, tokens_s: float = 0.0,
        truncated: bool = False,
    ) -> None:
        with self._lock:
            if not endpoint_id:
                return
            ep = self._state["endpoints"].get(endpoint_id)
            if not ep:
                # Endpoint from live traffic we've never seen (not in the template).
                # Add it with a neutral base; it's still worth tracking.
                key = endpoint_id
                host, port, model = self._parse_endpoint_id(endpoint_id)
                ep = self._state["endpoints"].setdefault(key, {
                    "base": {"host": host, "port": port, "model": model,
                             "id": endpoint_id, "grade": None, "tps": 0,
                             "supports_tools": True, "context_length": None,
                             "task_capabilities": ["code", "tool_calling"]},
                    "learned": None,
                    "task_scores": {},
                })

            # Update overall learned. Initialize the dict on first observation;
            # `learned_score()` returns None until sample_count > 0, so a bare
            # bootstrap doesn't dilute the base score.
            learned = ep.get("learned")
            if learned is None:
                learned = ep["learned"] = {
                    "success_rate": 0.5, "tokens_s": 50.0,
                    "avg_latency_ms": 300, "success_count": 0, "fail_count": 0,
                    "sample_count": 0,
                }
            if ok:
                learned["success_count"] = learned.get("success_count", 0) + 1
            else:
                learned["fail_count"] = learned.get("fail_count", 0) + 1
            learned["sample_count"] = learned.get("sample_count", 0) + 1
            total = learned["success_count"] + learned.get("fail_count", 0)
            learned["success_rate"] = learned["success_count"] / max(1, total)
            if latency_ms > 0:
                learned["avg_latency_ms"] = sc.update_ema(
                    learned.get("avg_latency_ms") or 300, latency_ms)
            if tokens_s > 0:
                learned["tokens_s"] = sc.update_ema(
                    learned.get("tokens_s") or 50.0, tokens_s)
            # Update truncation tracking
            if truncated:
                curr_trunc = learned.get("truncation_ema", 0.0)
                learned["truncation_ema"] = sc.update_ema(curr_trunc, 1.0)
            else:
                curr_trunc = learned.get("truncation_ema", 0.0)
                learned["truncation_ema"] = sc.update_ema(curr_trunc, 0.0)

            # Update task-level learned.
            ts = ep.setdefault("task_scores", {}).setdefault(task_id, {
                "success_rate": 0.5, "tokens_s": learned.get("tokens_s", 50.0),
                "avg_latency_ms": learned.get("avg_latency_ms", 300),
                "success_count": 0, "fail_count": 0, "sample_count": 0,
            })
            if ok:
                ts["success_count"] = ts.get("success_count", 0) + 1
            else:
                ts["fail_count"] = ts.get("fail_count", 0) + 1
            ts["sample_count"] = ts.get("sample_count", 0) + 1
            ts_total = ts["success_count"] + ts.get("fail_count", 0)
            ts["success_rate"] = ts["success_count"] / max(1, ts_total)
            if latency_ms > 0:
                ts["avg_latency_ms"] = sc.update_ema(ts.get("avg_latency_ms") or 300, latency_ms)
            if tokens_s > 0:
                ts["tokens_s"] = sc.update_ema(ts.get("tokens_s") or 50.0, tokens_s)

            self._dirty = True

    def _maybe_save(self) -> None:
        now = time.time()
        if now - self._last_save < self._SAVE_PERIOD:
            return
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
            snapshot = self._state
        try:
            st.save_state(snapshot)
            self._last_save = time.time()
        except Exception:
            logger.debug("could not persist model_routing state", exc_info=True)

    @staticmethod
    def _parse_endpoint_id(key: str) -> tuple[str, str, str]:
        # key looks like "1.2.3.4:11434/some/model:name"
        host_port, _, model = key.partition("/")
        host, _, port = host_port.rpartition(":")
        return host, port, model

    # ── hook adapter ──────────────────────────────────────────────────────────
    # The plugin's `register(ctx)` in __init__.py calls ctx.register_hook with
    # a closure that binds self. The hook callback receives kwargs from the hook
    # payload (model, provider, base_url, session_id, api_request_id, ...).
    #
    # We match live traffic to a catalog endpoint by looking up
    # `base_url` (host:port) + `model` in self._state["endpoints"]. If we
    # don't have that endpoint recorded, we fall back to just using the
    # live (host, model) pair we've just parsed.
    def _endpoint_id_for_call(self, kwargs: Mapping[str, Any]) -> Optional[str]:
        base_url = str(kwargs.get("base_url") or "")
        model = str(kwargs.get("model") or "")
        host_port = re.sub(r"^https?://", "", base_url).rstrip("/").rstrip("/v1")
        if not host_port and not model:
            return None
        key = f"{host_port}/{model}".rstrip("/")
        with self._lock:
            if key in self._state["endpoints"]:
                return key
        return key  # still emit it; _observe() will create the entry if needed

    # ── exposed for register() to wire into hooks ────────────────────────────
    def observe_post_api_request(self, **kwargs: Any) -> None:
        try:
            endpoint_id = self._endpoint_id_for_call(kwargs)
            if not endpoint_id:
                return
            error = kwargs.get("error")
            truncated = False
            if not error:
                # Detect stream truncation
                # A truncated stream = response exists but was cut short
                # (common signals: finish_reason != "stop" AND usage.output_tokens
                #  is near the max, OR response is shorter than expected, OR
                #  the client reported a timeout/EOF.)
                response = kwargs.get("response") or {}
                finish_reason = response.get("finish_reason") if isinstance(response, dict) else None
                # If we got a response but it was truncated, flag it
                if finish_reason and finish_reason not in ("stop", "end_turn"):
                    truncated = True

            ok = (error is None) and not truncated
            latency_ms = float(kwargs.get("api_duration") or 0) * 1000.0
            usage = kwargs.get("usage") or {}
            out_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            tokens_s = (out_tokens / (latency_ms / 1000.0)) if (latency_ms > 0 and out_tokens) else 0.0
            # We don't always know the task in post_api_request (that's the main loop)
            # so we treat it as a generic "code" observation.
            self._observe(
                endpoint_id, "code",
                ok=ok, latency_ms=latency_ms, tokens_s=tokens_s,
                truncated=truncated,
            )
            self._maybe_save()
        except Exception:
            logger.debug("model-router: observe_post_api_request failed", exc_info=True)

    def observe_post_auxiliary_call(self, **kwargs: Any) -> None:
        try:
            endpoint_id = self._endpoint_id_for_call(kwargs)
            if not endpoint_id:
                return
            ok = kwargs.get("error") is None
            latency_ms = float(kwargs.get("api_duration") or 0) * 1000.0
            usage = kwargs.get("usage") or {}
            out_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            tokens_s = (out_tokens / (latency_ms / 1000.0)) if (latency_ms > 0 and out_tokens) else 0.0
            aux_task = str(kwargs.get("aux_task") or "code")
            self._observe(endpoint_id, aux_task, ok=ok, latency_ms=latency_ms, tokens_s=tokens_s)
            self._maybe_save()
        except Exception:
            logger.debug("model-router: observe_post_auxiliary_call failed", exc_info=True)

    def observe_api_request_error(self, **kwargs: Any) -> None:
        try:
            endpoint_id = self._endpoint_id_for_call(kwargs)
            if not endpoint_id:
                return
            latency_ms = float(kwargs.get("api_duration") or 0) * 1000.0
            aux_task = str(kwargs.get("aux_task") or "code")
            self._observe(endpoint_id, aux_task, ok=False, latency_ms=latency_ms, tokens_s=0.0)
            self._maybe_save()
        except Exception:
            logger.debug("model-router: observe_api_request_error failed", exc_info=True)


def register(ctx) -> None:
    provider = ModelRouterProvider()
    ctx.register_delegation_provider(provider)
    ctx.register_hook("post_api_request", provider.observe_post_api_request)
    ctx.register_hook("post_auxiliary_call", provider.observe_post_auxiliary_call)
    ctx.register_hook("api_request_error", provider.observe_api_request_error)

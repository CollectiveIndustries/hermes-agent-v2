"""Model Router delegation backend — learns which endpoints are best for which tasks.

Reads the user's template JSON (llama_out.json) at startup to build a base score
per endpoint x task. Observes `post_api_request`, `post_auxiliary_call`, and
`api_request_error` hooks to learn actual success/latency/tokens_s per
endpoint x task. Routes each `delegate_task` to the best-scoring endpoint.

State lives at ~/.hermes/model_routing_state.json (portable, zero deps).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def _hermes_home() -> Path:
    """Resolve the Hermes home directory (profile-aware)."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())


def state_path() -> Path:
    """Absolute path to the learned state JSON file."""
    return _hermes_home() / "model_routing_state.json"


def template_path() -> Path:
    """Absolute path to the user's template JSON file (llama_out.json)."""
    # Look for llama_out.json in a few likely locations
    for candidate in (
        _hermes_home() / "llama_out.json",
        Path.home() / "llama_out.json",
        Path("/home/admiral/llama_out.json"),
    ):
        return candidate  # Return the first one (caller will handle missing)
    return _hermes_home() / "llama_out.json"


def load_template(json_path: Optional[str] = None) -> dict[str, Any]:
    """Load the user's template JSON. Returns a normalized dict with an
    `endpoints` key mapping `host:port/model` → base score info.

    Accepts either:
      a. The raw ralph-provider-catalog/v1 schema (with `models: [...]`)
      b. A simplified per-endpoint map (with `endpoints: {...}`)

    Returns:
        {
          "endpoints": {
            "<host:port>/<model>": {
              "host": ..., "port": ..., "model": ...,
              "grade": "A"|"B"|..."F"|null,
              "tps": float,
              "supports_tools": bool,
              "context_length": int|null,
              "task_capabilities": [...],
            }, ...
          },
          "generated_at": str|null,
        }
    """
    path = Path(json_path) if json_path else template_path()
    if not path.exists():
        logger.debug("template %s not found", path)
        return {"endpoints": {}, "generated_at": None}

    try:
        raw = json.loads(path.read_text())
    except Exception:
        logger.exception("could not parse template %s", path)
        return {"endpoints": {}, "generated_at": None}

    # Raw ralph-provider-catalog/v1 schema
    if isinstance(raw, dict) and "models" in raw and isinstance(raw["models"], list):
        return _normalize_ralph_catalog(raw)

    # Already in simplified shape
    if isinstance(raw, dict) and "endpoints" in raw:
        return raw

    # A single record (dict with host/port/model)
    if isinstance(raw, dict) and "model" in raw and "host" in raw:
        return {"endpoints": {raw["id"]: _normalize_record(raw)}, "generated_at": None}

    logger.warning("unrecognized template shape at %s", path)
    return {"endpoints": {}, "generated_at": None}


def _normalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    host = rec.get("host", "")
    port = rec.get("port", "")
    model = rec.get("model", "")
    caps = rec.get("capabilities") or []
    return {
        "host": host,
        "port": port,
        "model": model,
        "id": rec.get("id") or f"{host}:{port}/{model}",
        "grade": rec.get("coding_grade"),
        "tps": rec.get("median_tokens_per_second"),
        "supports_tools": bool("tools" in caps),
        "context_length": rec.get("context_length"),
        "task_capabilities": rec.get("task_capabilities") or _infer_task_caps(caps),
        "family": rec.get("family"),
        "param_size": rec.get("parameter_size"),
    }


def _infer_task_caps(caps: list[str]) -> list[str]:
    """Map a `capabilities` list to a `task_capabilities` list."""
    out = []
    caps_l = {c.lower() for c in caps}
    if "completion" in caps_l:
        out.extend(["code", "title", "compression", "vision", "approval"])
    if "tools" in caps_l:
        out.extend(["tool_calling", "code_generation"])
    return sorted(set(out))


def _normalize_ralph_catalog(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert ralph-provider-catalog/v1 into our normalized shape."""
    endpoints: dict[str, dict[str, Any]] = {}
    for rec in raw.get("models", []):
        base = _normalize_record(rec)
        key = base["id"]
        # If we've seen this endpoint, merge: take the best grade, max tps,
        # union of task_capabilities
        if key in endpoints:
            prev = endpoints[key]
            if (rec.get("coding_grade") or "").upper() < str(prev.get("grade") or "Z"):
                prev["grade"] = rec.get("coding_grade")
            prev_tps = prev.get("tps") or 0
            if (rec.get("median_tokens_per_second") or 0) > prev_tps:
                prev["tps"] = rec.get("median_tokens_per_second")
            if rec.get("context_length"):
                prev["context_length"] = max(prev.get("context_length") or 0, rec["context_length"]) or rec["context_length"]
            if "tools" in (rec.get("capabilities") or []):
                prev["supports_tools"] = True
            for cap in _infer_task_caps(rec.get("capabilities") or []):
                if cap not in prev["task_capabilities"]:
                    prev["task_capabilities"].append(cap)
        else:
            endpoints[key] = base

    return {"endpoints": endpoints, "generated_at": raw.get("generated_at")}


def load_state() -> dict[str, Any]:
    """Load learned state or return an empty scaffold."""
    path = state_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            logger.exception("could not read state %s", path)
    return {
        "version": 1,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "endpoints": {},
    }


def save_state(state: dict[str, Any]) -> None:
    """Atomically write the learned state to disk."""
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now_iso()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

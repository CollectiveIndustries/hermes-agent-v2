# Model Router Plugin — Design

## Overview
A backend plugin that provides intelligent, task-centric model routing for HermeticAgent.
Learns endpoint performance over time using the JSON template as a prior.

## Plugin Kind
**Backend** (like image_gen, video_gen, browser provider, etc.)
Selected via `delegation.backend: model-router` in config.yaml.

## Public API

### RouteTask (main entry point)
```python
def route_for_task(task: str, context: dict) -> dict:
    """
    task: "code" | "tool_calling" | "code_generation" | "title" | "compression" | "vision"
    context: {
        "priority": "high" | "normal" | "low",
        "max_tokens": 4096,
        "requires_tools": true/false,
        "preferred_family": "qwen3moe" | null
    }
    returns: {
        "provider": "custom",
        "model": "huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M",
        "base_url": "http://180.159.55.50:11434/v1",
        "reasoning_effort": ""
    }
    """
```

### Learning Hooks
- `post_api_request` → extract success/failure + latency + tokens/s → update score
- `post_auxiliary_call` → auxiliary task learning
- `api_request_error` → failure signal

## Data Models

### EndpointState
```json
{
  "base": {"grade": "A", "tps": 49.0, "supports_tools": true, "context": 262144},
  "learned": {
    "success_rate": 0.95,
    "avg_latency_ms": 230,
    "avg_tokens_s": 42.5,
    "failures": 5,
    "sample_count": 200
  },
  "task_scores": {
    "code": {"success": 0.97, "latency": 180, "sample": 150},
    "tool_calling": {"success": 0.88, "latency": 450, "sample": 80},
    "compression": {"success": 0.75, "latency": 500, "sample": 40}
  }
}
```

### Scoring Algorithm
```
for each endpoint:
  base_score = 0.5 * grade_weight(code_grade) + 0.3 * speed_weight(tps) + 0.2 * tool_weight(supports_tools)
  
  if learned_scores:
    empirical_score = 0.7 * success_rate + 0.3 * speed_weight(learned_tokens_s)
  
  final_score = 0.6 * base_score + 0.4 * empirical_score (if empirical exists, else base_score)

Select endpoint with max(final_score) for the given task + context
```

## State Persistence
File: `~/.hermes/model_routing_state/v1.json`
Structure:
```json
{
  "version": 1,
  "updated_at": "2026-11-22T04:00:00Z",
  "endpoints": {
    "180.159.55.50:11434/qwen3-coder": { ... EndpointState ... },
    "1.119.199.74:11434/llama3.2:1b": { ... EndpointState ... }
  }
}
```

## Configuration
```yaml
delegation:
  backend: model-router
  max_concurrent_children: 10
  
model_routing:
  enabled: true
  template: "~/.hermes/model_routing_templates/llama_out.json"
  state_version: 1
  learning_rate: 0.1
```

## Bootstrap
1. Read template JSON → build `base_score` for each endpoint
2. Initialize state file (or load if exists)
3. For each endpoint, seed `learned.*` from template if no prior data
4. Start observing hooks

## Learning Loop (on each hook fire)
1. Extract `endpoint_id` (host:port/model)
2. Extract `task_type` (from `aux_task` or from tool name)
3. Update `success_rate`, `avg_latency_ms`, `avg_tokens_s` via EMA
4. Increment counters
5. Write state file (throttled, e.g. every 10s to avoid disk thrashing)
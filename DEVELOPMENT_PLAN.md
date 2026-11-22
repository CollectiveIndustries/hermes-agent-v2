# Hermes Dynamic Model Routing Plan

## Summary

Based on the JSON (158 records, 54 unique endpoints), this document proposes:
1. One primary A-class coding model selected by maximum tokens/s among all tool-capable A-grade models  
2. Per-task auxiliary model routing (based on task's speed and capability requirements)
3. Fallback chaining for the main model
4. A plugin framework design for runtime model ranking

## Analysis of A-Class Providers

The highest-rated endpoint is at host `180.159.55.50:11434` with model 
`huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M`, scoring:
- A grade (highest)
- Has tools capability
- 262K context window
- 49 tokens/s
- Family: qwen3moe, 30.5B parameters, Q4_K_M quantization

The top A-class endpoints (by speed) are:
1. `180.159.55.50:11434/huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M` (~49 t/s)
2. `136.243.57.187:11434/huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M` (~16.9 t/s) 
3. `43.142.105.181:11434/huihui_ai/qwen2.5-abliterate:7b` (~3.2 t/s)

## Proposed Hermetic Config for Main Model 

```yaml
model:
  provider: custom
  default: "huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M"
  base_url: "http://180.159.55.50:11434/v1"
  api_key: ollama
```

## Per-Task Auxiliary Model Routs

| Task | Endpoint | Grade | Tokens/s | Context | Reason |
|------|----------|-------|----------|---------|--------|
| vision | 1.119.199.74:11434/lab_canary_w1_1_119_199_74_102412:latest | C | 162.6 | 131072k | Fastest available model with tools |
| title | 1.119.199.74:11434/llama3.2:1b | C | 161.2 | 131072k | Fastest available with tools |
| compression | 136.243.57.187:11434/qwen2.5:0.5b | C | 71.2 | 32768k | Good balance, not too slow |
| approval | 180.159.55.50:11434/huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M | A | 49.0 | 262144k | Uses main model for high-fidelity analysis |

## Fallback Chain

Primary model has a fallback chain:
- `136.243.57.187:11434/huihui_ai/qwen3-coder-abliterated:30b-a3b-instruct-q4_K_M`
- `43.142.105.181:11434/huihui_ai/qwen2.5-abliterate:7b` 
- `1.119.199.74:11434/lab_canary_w1_1_119_199_74_102412:latest`

## Proposed Plugin Architecture

Implement dynamic model ranking as a plugin:
1. `model_routing.py` - Plugin that monitors endpoint health, measures real-time performance
2. `endpoint_probe.py` - Runtime HTTP probes to measure token/s on all endpoints  
3. `ranking_cache.py` - Caches rankings and invalidates on change

## Dynamic Behavior Pattern

For each task, select the model based on:
1. **Context needs** (what is required for quality)
2. **Speed requirements** (how quickly must it respond)  
3. **Availability** (can we reliably contact that endpoint?)
4. **Task-type** (some tasks prefer fast, others need more depth)

This enables runtime switching between the best models of each type:
- Primary: A-class with high speed for coding
- Auxiliary: Lower-tier for fast support work 
- Fallback: Alternatives when primary is unavailable or slower than expected

## Configuration Schema

```yaml  
smart_model_routing: 
  # Auto-detect from catalog on first run (static)
  enabled: true  

delegation:
  model: "openrouter/anthropic/claude-opus-4.8"  # For subagents
  provider: "openrouter"
  
auxiliary:  # Per-task routing - these default to auto for now
  vision: 
    model: ""  # auto (uses primary's main model)
    provider: ""
    base_url: ""
    timeout: 30
  title_generation:
    provider: "custom"  # fast, cheap endpoint  
    model: "qwen2.5:0.5b"
    base_url: "http://136.243.57.187:11434/v1"
    timeout: 30
  compression:
    provider: "custom" 
    model: "qwen2.5:0.5b"
    base_url: "http://136.243.57.187:11434/v1"
    timeout: 120  
```

This gives the user a starting point with static selection that can be upgraded to full dynamic routing.
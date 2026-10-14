# Vendored from Cloud-OpsBench

Copied unchanged from https://github.com/LLM4Ops/Cloud-OpsBench at commit `ea05daf`
(MIT License, Copyright (c) 2026 Yilun Wang; see `LICENSE` in this folder).

| File here | Upstream path | What it is |
|---|---|---|
| `snapshot.py` | `agents/cloudops_agent/tools/snapshot.py` | `KubernetesTools`: replays every tool from a case's `tool_cache.json` |
| `cloudops.py` | `agents/cloudops_agent/tools/cloudops.py` | `create_k8s_tools()`: the tool list (name, description, pydantic args schema) per system / category |
| `contracts.py` | `agents/cloudops_agent/runtime/contracts.py` | `build_expected_output()`: the Submit format, valid root causes and resource names |
| `system_prompt.md` | `agents/cloudops_agent/harness/system_prompt.md` | Their system prompt |

Do not edit these files: keeping them identical is what makes Brownout's numbers comparable
with the paper's. Brownout-specific code lives one folder up.

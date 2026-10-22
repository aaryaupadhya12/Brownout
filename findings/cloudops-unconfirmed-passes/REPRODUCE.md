# Reproduce

## Pinned inputs

| Input | Value |
|---|---|
| Cloud-OpsBench | `https://github.com/LLM4Ops/Cloud-OpsBench`, commit `ea05daf`; set `CLOUDOPSBENCH_ROOT` |
| Brownout | this repo, branch `claude/interface-v1` (commit the files before quoting a hash) |
| Saved runs | `runs/*__*.json`, 1,097 files (Nemotron-3-Nano-30B and DeepSeek-V4-Flash via Nebius) |
| Strict verifier | v0.2.0, output name hash `6f0952b5` (covers `strict.py`, `verifier.py`, `process_eval.py`) |
| Judge | `zai-org/GLM-5.3-Flash` via Nebius, temperature 0, prompt v1 |

## Free steps (no API calls)

```powershell
python -m unittest tests.cloudops_strict_test tests.cloudops_audit_judge_test -v      # 31 tests
python -m src.envs.SystemEnv.cloudops.regrade_strict                                  # -> runs/regrade/strict-6f0952b5.jsonl
python -I findings/cloudops-unconfirmed-passes/compute_numbers.py                     # compare with expected.json
```

Claim A and C need only these. If the output file name has a different hash, the verifier code
differs from the one these numbers came from.

## Free step: legitimate control (expert paths, claims A and C)

```powershell
python -m src.envs.SystemEnv.cloudops.golden_control --write   # runs/regrade/golden-control.jsonl
python -I findings/cloudops-unconfirmed-passes/compute_numbers.py --write
```

Reads `golden-trajectory/`, `process-label/` and `benchmark/*/metadata.json` from
CLOUDOPSBENCH_ROOT. Refuses to overwrite an existing output. Expected under `"golden_control"`:
151 of 1,100 boutique paths rejected raw, 111 of those rescued by pod-as-config, 80 of 89
unconfirmed runs where every expert path passes.

## Paid step (Claim B)

```powershell
python -m src.envs.SystemEnv.cloudops.sample_unconfirmed runs/regrade/strict-6f0952b5.jsonl --seed 2 --n 1000
python -m src.envs.SystemEnv.cloudops.audit_judge runs/regrade/strict-6f0952b5-audit-seed2.csv --max-calls 120
```

Reverse direction (the 73 rescued runs) and its negative control:

```powershell
python -m src.envs.SystemEnv.cloudops.sample_unconfirmed runs/regrade/strict-6f0952b5.jsonl --rescued --seed 0 --n 1000
python -m src.envs.SystemEnv.cloudops.audit_judge runs/regrade/strict-6f0952b5-rescued-seed0.csv --mode rescued --max-calls 100
python findings/cloudops-unconfirmed-passes/negative_control.py          # 12 calls
python -I findings/cloudops-unconfirmed-passes/compute_numbers.py --write
```

Forward census: about 89 calls of up to ~29k characters each; reverse: 73 calls; control: 12. failed replies are retried on a rerun, parsed
verdicts are cached (`runs/regrade/judge-cache-zai-org_GLM-5.3-Flash.jsonl`), so a rerun with the
cache costs nothing. Temperature 0 does not guarantee identical labels across provider versions;
expect small differences without the cache.

## Expected

`expected.json`. Key values: Nemotron unconfirmed 89 / 322; DeepSeek 0 / 468; census labelled
82 of 89 (57 weak, 21 strong, 3 label_gap, 1 blind).

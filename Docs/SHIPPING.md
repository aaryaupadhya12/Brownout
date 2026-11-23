# How Brownout ships

The standard every change and every release meets. Modelled on how production Kubernetes agents
(e.g. NudgeBee) are built: read freely, change only on approval, declare the plan and run exactly
that plan, and check health before and after. `scripts/ship_check.py` enforces the mechanical
parts on every commit; the rest is checked in review.

## Six rules

1. **Plan, then run exactly the plan.** Anything that spends money or changes a machine (GPU
   boxes, paid model calls, training) prints its plan first: commands, GPUs, time and cost
   estimate. Dry-run is the default; it runs only with `--yes`. It runs only the commands it
   declared.
2. **Read freely, change on approval.** Agents and environments can read anything in their
   sandbox. Grader, verifier and reward changes need review before they merge, because they change
   what every past and future number means.
3. **Health check before and after.** Every environment ships with two checks that run before and
   after any change: the reference solution (oracle) scores 1, and an empty answer (nop) scores 0.
   A release runs the full check list on a clean machine before and after the tag.
4. **One failure, one cause.** Infra errors and model failures are separate: a crashed scorer or
   container is `ok=False`, never reward 0. Failures are grouped by cause, not listed one per run.
5. **Guardrails on every run.** Every run has budgets (turns, tokens, seconds, dollars) and stops
   itself at the first one it hits. No run without a cost cap.
6. **Your machine, your keys.** Keys come from the environment, never from files in the repo.
   Nothing phones home. Saved runs are records and are never edited.

## Every commit (enforced by `scripts/ship_check.py`, locally and in CI)

- Offline tests pass with the network blocked: `python scripts/test_offline.py`.
- No run outputs or datasets are committed: `runs/`, `jobs/`, `lcb/`, `data/`.
- No secrets in the diff (API keys, private keys, `.env` files) and no file over 5 MB.
- A new Python file under `src/`, `cli/` or `train/` is under 100 lines, or its first lines
  contain `# over-100: <reason>`. Existing long files get a warning when changed.

- CI (`.github/workflows/ci.yml`) runs `ship_check.py --all` on every push and pull request,
  from `uv.lock`, with actions pinned by commit SHA and read-only permissions.
- Dependencies change only through `pyproject.toml` + `uv lock`; then re-export
  `requirements.txt` with `uv export --locked --no-hashes --format requirements-txt -o requirements.txt`.

## Every environment

- `task.toml` has timeouts and resource limits (CPU, memory, storage).
- Oracle = 1 and nop = 0 recorded in the env's README, with the Harbor job folder name.
- The reward reads only the task and the trace (see CLAUDE.md design rules).
- One command reproduces a run, written in the README.
- Its data is pinned to an exact commit, and saved runs replay on it with no model
  (Cloud-OpsBench: `cloudops.data check`, then `cloudops.replay`; any difference blocks a release).

## Every release

- Version pinned everywhere it appears (package, CLI `--version`, run cards).
- `CHANGELOG.md` entry: what changed, what broke, what was not verified.
- Clean-machine install from the README alone, commands copied exactly.
- Every number in the README or on the website traces to a file in the repo.
- Tag only after all of the above pass. Known gaps are written down, not hidden.

## Install the check

```bash
uv sync --locked                      # the pinned environment (Python 3.12)
git config core.hooksPath .githooks   # runs scripts/ship_check.py before every commit
python scripts/ship_check.py          # by hand; --no-tests skips tests, --all checks every file
```

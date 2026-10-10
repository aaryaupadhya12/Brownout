# Running Brownout

## Requirements

- Python 3.10 or newer
- The `openai` Python package for the OpenAI-compatible client and its tests

`requirements.txt` is currently only a placeholder, so install `openai` explicitly.
Run these commands from the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install openai
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
in that PowerShell window, then activate the environment again.

## Run the tests

With the virtual environment active, run the repository's `unittest` tests:

```powershell
python -m unittest discover -s tests -p "*.py"
```

The test files are in `tests/`; the wildcard includes names such as `client_test.py`
and `reset_tests.py`.

## Live model runs

`raw_call.py` is the intended example entry point for making Nebius API calls:

```powershell
python raw_call.py
```

However, the current script contains a placeholder API key (`fake for now`) and
does not read a key from the environment. Do not put a real key in source control.
The live path also depends on the harness, which currently has unresolved runtime
issues, so a live run is not ready to use as-is. Configure secret handling and fix
the harness before relying on this command.

The harness and test fixtures are under `src/` and `tests/`. The model client
accepts an OpenAI-compatible base URL and model ID; see `Clients/openai_compact.py`
and `raw_call.py` when wiring up a provider.

"""Provider name -> (base_url, api key). The one place that reads secrets.

    configs/providers.toml   base_url + the NAME of the env var (committed, no secrets)
    .env / Modal secret       the VALUE (never committed)

    client = make_client("nebius", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")

Rules: a key is read only here, only when a client is made, and never written to a
Trajectory, a config, a log line or an error message (error messages name the env var).
"""

import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVIDERS_FILE = os.path.join(ROOT, "configs", "providers.toml")
ENV_FILE = os.path.join(ROOT, ".env")


def load_providers(path: str = PROVIDERS_FILE) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_env_file(path: str = ENV_FILE) -> None:
    """Read KEY=VALUE lines from a .env file into os.environ.

    - a missing file is fine (Modal and CI set real env vars instead): just return
    - skip blank lines and lines starting with '#'
    - strip spaces around key and value, and one pair of surrounding quotes from the value
    - never overwrite a variable that is already set: the real environment wins
    """
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")        # "A = 1" -> "A ", "=", " 1"
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]                    # "two words" -> two words
        os.environ.setdefault(key, value)          # only sets it if not already set


def resolve(provider: str, providers: dict | None = None) -> tuple[str, str]:
    """(base_url, api_key) for `provider`.

    - unknown provider -> KeyError listing the known ones (like registry.get)
    - api_key_env == "" -> no key needed: return "EMPTY" (what OpenAI-compatible servers expect)
    - env var not set or empty -> RuntimeError naming the VARIABLE, e.g.
          "provider 'nebius' needs NEBIUS_API_KEY: set it in .env or the environment"
      The message must never contain a key value.
    """
    providers = providers if providers is not None else load_providers()
    p = providers.get(provider)
    if p is None:
        raise KeyError(f"unknown provider {provider!r}; known: {', '.join(providers)}")
    var = p["api_key_env"]
    if var == "":
        return p["base_url"], "EMPTY"
    key = os.environ.get(var)
    if not key:
        raise RuntimeError(f"provider {provider!r} needs {var}: set it in .env or the environment")
    return p["base_url"], key


def make_client(provider: str, model_id: str, **kwargs):
    """An OpenAICompatClient for `provider`. kwargs go to the client (min_gap_s, ...)."""
    load_env_file()
    base_url, key = resolve(provider)
    from Clients.openai_compact import OpenAICompatClient   # needs openai: imported only here
    return OpenAICompatClient(model_id, base_url, key, **kwargs)

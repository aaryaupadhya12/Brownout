"""Fetch PrimeIntellect/Reverse-Text-RL without new dependencies, and cache it.

Hugging Face serves dataset rows as JSON (no `datasets` package needed):

    https://datasets-server.huggingface.co/rows?dataset=PrimeIntellect/Reverse-Text-RL
        &config=default&split=train&offset=0&length=100

Response shape (checked 2026-10-22: 1,000 rows, one field):
    {"features": [{"name": "prompt", ...}], "num_rows_total": 1000,
     "rows": [{"row_idx": 0, "row": {"prompt": "The community in Bruck was merged into it"}}, ...]}

`length` is capped at 100 per request, so 1,000 rows = 10 requests. Fetch once, then always
read the cache: tests and runs must not depend on the network.
"""

import json
import os

DATASET = "PrimeIntellect/Reverse-Text-RL"
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100
DEFAULT_CACHE = os.path.join("data", "reverse_text", "train.json")


def fetch_rows(cache_path: str = DEFAULT_CACHE, dataset: str = DATASET,
               split: str = "train") -> list[dict]:
    """Return [{"prompt": str}, ...] in dataset order.

    If cache_path exists: read it and return (no network).
    Else: page through ROWS_API with offset=0,100,200,... until num_rows_total, keep only
    row["row"], write the list to cache_path (create the folder), return it.

    Think about: what if the download fails halfway? (Write the cache only after every page
    arrived, or a later run reads a half cache and silently has 300 tasks.) Use urllib from
    the standard library; set a timeout.
    """
    raise NotImplementedError("TODO(you): fetch_rows")

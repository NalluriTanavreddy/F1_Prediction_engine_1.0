"""Cached fetch helper for the Ergast-compatible F1 API (Jolpica).

Adapted from BoxBox (https://github.com/NalluriTanavreddy/boxbox)
src/boxbox/utils/ergast.py. BoxBox is an MCP server answering live
queries, so it uses a short in-memory TTL cache. This is a batch
ingestion script that re-runs repeatedly during development against
mostly-immutable historical data, so it swaps that for a persistent
on-disk JSON cache under data/raw/ instead — that also doubles as the
"cache raw API responses locally" requirement.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from f1_predict.clients.http import get_client

# ergast.com shut down after the 2024 season; Jolpica is the drop-in
# replacement with identical paths and JSON structure.
ERGAST_BASE_URL = os.getenv("F1_PREDICT_ERGAST_BASE_URL", "https://api.jolpi.ca/ergast/f1")

# Jolpica caps limit at 100; the default of 30 truncates full-season result sets.
DEFAULT_LIMIT = 100

RAW_DATA_DIR = Path(os.getenv("F1_PREDICT_RAW_DATA_DIR", "data/raw"))

# Jolpica's public tier rate-limits fairly aggressively and returns 429s in
# bursts. Throttle every *actual* network call (cache hits skip this) to one
# request per MIN_REQUEST_INTERVAL, and back off on 429 using Retry-After.
MIN_REQUEST_INTERVAL = 0.5
MAX_RETRIES = 6

_last_request_at = 0.0
_throttle_lock = asyncio.Lock()


async def _throttled_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    global _last_request_at
    for attempt in range(MAX_RETRIES):
        async with _throttle_lock:
            wait = _last_request_at + MIN_REQUEST_INTERVAL - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            _last_request_at = time.monotonic()
            response = await client.get(url, params=params)

        if response.status_code != 429:
            return response

        retry_after = float(response.headers.get("Retry-After", 1.0))
        await asyncio.sleep(retry_after)

    response.raise_for_status()
    return response


def _cache_path(path: str) -> Path:
    return RAW_DATA_DIR / f"{path}.json"


def _has_no_races(data: dict) -> bool:
    """True if a RaceTable/StandingsTable response carries zero entries.

    A round whose session hasn't happened yet returns an empty table. We
    must not let that empty response poison the disk cache forever, or a
    round that completes between two runs would stay "missing" forever.
    """
    race_table = data.get("MRData", {}).get("RaceTable")
    if race_table is not None:
        return not race_table.get("Races")
    return False


async def ergast_get(
    path: str,
    limit: int = DEFAULT_LIMIT,
    *,
    force_refresh: bool = False,
    cache_empty: bool = False,
) -> dict:
    """Fetch an Ergast API path (without the .json suffix) as a parsed dict.

    Responses are cached indefinitely to disk under data/raw/<path>.json,
    keyed only by path (not by limit — this project always requests full
    result sets).

    By default an empty RaceTable (a round whose session hasn't happened
    yet) is never cached, so a round that completes between two ingestion
    runs doesn't stay stuck "missing". Pass cache_empty=True for endpoints
    where "no data" is itself a permanent, known fact once the round is
    otherwise complete — e.g. a non-sprint weekend's /sprint.json legitimately
    and permanently returns no races, and re-fetching that every run just to
    re-confirm "still no sprint" wastes API calls. Callers are responsible
    for only setting it once they know the round has actually happened.
    """
    cache_file = _cache_path(path)
    if not force_refresh and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cache_empty or not _has_no_races(cached):
            return cached

    url = f"{ERGAST_BASE_URL}/{path}.json"
    client = get_client()
    response = await _throttled_get(client, url, {"limit": limit})
    response.raise_for_status()
    data = response.json()

    if cache_empty or not _has_no_races(data):
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data

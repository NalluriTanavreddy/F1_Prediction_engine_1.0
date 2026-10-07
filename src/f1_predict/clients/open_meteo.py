"""Cached fetch helpers for Open-Meteo weather, adapted the same way as
clients/ergast.py was adapted from BoxBox (https://github.com/NalluriTanavreddy/boxbox
src/boxbox/tools/circuits.py get_weather_at_circuit) — same forecast endpoint
and daily variables, but with a persistent on-disk cache instead of BoxBox's
TTL cache, since this is a batch job re-run during development.

Two deliberately separate functions, for two different situations:

- fetch_historical_weather(): Open-Meteo's *archive* API — actual observed
  conditions for a past date. Used to backfill weather for every race
  already in the dataset. This has no "rain probability" field (it's
  reanalysis of what happened, not a forecast), only actual precipitation.
- fetch_forecast_weather(): Open-Meteo's *forecast* API — a real pre-race
  forecast, for a date that hasn't happened yet. This is what a live
  prediction must use, since actual weather obviously isn't known yet.

Mixing these up would be a real leakage bug (training on omniscient
hindsight weather is a documented, accepted tradeoff; silently serving
hindsight weather at inference time would not be), so callers must pick
one explicitly — there's no single "get_weather" that guesses.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

from f1_predict.clients.http import get_client

ARCHIVE_BASE_URL = os.getenv("F1_PREDICT_OPENMETEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive")
FORECAST_BASE_URL = os.getenv("F1_PREDICT_OPENMETEO_FORECAST_URL", "https://api.open-meteo.com/v1/forecast")

RAW_DATA_DIR = Path(os.getenv("F1_PREDICT_RAW_DATA_DIR", "data/raw")) / "weather"

# Open-Meteo is far more lenient than Jolpica, but a small throttle costs
# nothing and avoids ever tripping a rate limit during a full backfill.
MIN_REQUEST_INTERVAL = 0.2
MAX_RETRIES = 5

_last_request_at = 0.0
_throttle_lock = asyncio.Lock()

# Daily aggregates, not hourly: a race's exact start hour varies by weekend
# and timezone handling is fiddly. "Race-day" granularity (max temp, total
# precip, max wind over the day) matches what the task asked for and is
# what the historical archive and forecast APIs can both provide uniformly.
HISTORICAL_DAILY_VARS = "temperature_2m_max,precipitation_sum,windspeed_10m_max"
FORECAST_DAILY_VARS = "temperature_2m_max,precipitation_probability_max,precipitation_sum,windspeed_10m_max"


async def _throttled_get(client: httpx.AsyncClient, url: str, params: dict) -> httpx.Response:
    global _last_request_at
    for _ in range(MAX_RETRIES):
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


def _cache_path(kind: str, latitude: float, longitude: float, date: str) -> Path:
    return RAW_DATA_DIR / kind / f"{latitude:.4f}_{longitude:.4f}_{date}.json"


async def _cached_get(cache_file: Path, url: str, params: dict) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    client = get_client()
    response = await _throttled_get(client, url, params)
    response.raise_for_status()
    data = response.json()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _first_daily_value(data: dict, var: str) -> float | None:
    values = data.get("daily", {}).get(var, [])
    return values[0] if values and values[0] is not None else None


async def fetch_historical_weather(latitude: float, longitude: float, date: str) -> dict:
    """Actual observed race-day weather for a past date, from Open-Meteo's archive API.

    Returns temp_max_c, precip_mm, wind_speed_max_kph, rain_probability_pct
    (always None — reanalysis has no probability field), and source="historical_actual".
    """
    cache_file = _cache_path("historical", latitude, longitude, date)
    data = await _cached_get(
        cache_file,
        ARCHIVE_BASE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "daily": HISTORICAL_DAILY_VARS,
            "timezone": "auto",
        },
    )
    return {
        "temp_max_c": _first_daily_value(data, "temperature_2m_max"),
        "precip_mm": _first_daily_value(data, "precipitation_sum"),
        "wind_speed_max_kph": _first_daily_value(data, "windspeed_10m_max"),
        "rain_probability_pct": None,
        "source": "historical_actual",
    }


async def fetch_forecast_weather(latitude: float, longitude: float, date: str) -> dict:
    """A real pre-race forecast for a future date, from Open-Meteo's forecast API.

    Open-Meteo only forecasts ~16 days out; a date further out than that
    makes the API itself reject the request (HTTPStatusError via
    raise_for_status), which propagates to the caller. Call this close to
    the race weekend, not far in advance.
    """
    cache_file = _cache_path("forecast", latitude, longitude, date)
    data = await _cached_get(
        cache_file,
        FORECAST_BASE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": date,
            "end_date": date,
            "daily": FORECAST_DAILY_VARS,
            "timezone": "auto",
        },
    )
    return {
        "temp_max_c": _first_daily_value(data, "temperature_2m_max"),
        "precip_mm": _first_daily_value(data, "precipitation_sum"),
        "wind_speed_max_kph": _first_daily_value(data, "windspeed_10m_max"),
        "rain_probability_pct": _first_daily_value(data, "precipitation_probability_max"),
        "source": "forecast",
    }

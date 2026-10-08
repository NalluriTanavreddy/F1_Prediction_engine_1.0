"""Shared test fixtures: a fake 2026 round 17 (Singapore) with 2 qualifiers.

2026-10-11 is chosen because it's after every date already in the
committed data/processed/race_dataset.parquet (latest round there is
round 16, 2026-10-04) — so tests exercise the real committed dataset as
history via the real load_history_before, with only the network layer
(ergast_get, fetch_forecast_weather) mocked. No synthetic history fixture
needed, and the pipeline is exercised against real data end to end.
"""

from unittest.mock import AsyncMock

import pytest

SEASON = 2026
ROUND_NO = 17
RACE_DATE = "2026-10-11"

SCHEDULE_ENTRY = {
    "season": str(SEASON),
    "round": str(ROUND_NO),
    "raceName": "Singapore Grand Prix",
    "date": RACE_DATE,
    "Circuit": {
        "circuitId": "marina_bay",
        "circuitName": "Marina Bay Street Circuit",
        "Location": {"lat": "1.2914", "long": "103.8640", "country": "Singapore"},
    },
}

QUALIFYING_RESULTS = [
    {
        "position": "1",
        "Driver": {"driverId": "max_verstappen", "code": "VER", "givenName": "Max", "familyName": "Verstappen"},
        "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
        "Q1": "1:29.000",
        "Q2": "1:28.500",
        "Q3": "1:28.000",
    },
    {
        "position": "2",
        "Driver": {"driverId": "hamilton", "code": "HAM", "givenName": "Lewis", "familyName": "Hamilton"},
        "Constructor": {"constructorId": "ferrari", "name": "Ferrari"},
        "Q1": "1:29.200",
        "Q2": "1:28.700",
        "Q3": "1:28.300",
    },
]

EMPTY_RACE_TABLE = {"MRData": {"RaceTable": {"Races": []}}}


def race_table(races: list) -> dict:
    return {"MRData": {"RaceTable": {"Races": races}}}


def schedule_response() -> dict:
    return race_table([SCHEDULE_ENTRY])


def qualifying_response() -> dict:
    return race_table([{**SCHEDULE_ENTRY, "QualifyingResults": QUALIFYING_RESULTS}])


FORECAST_WEATHER = {
    "temp_max_c": 30.0,
    "precip_mm": 1.5,
    "wind_speed_max_kph": 12.0,
    "rain_probability_pct": 40.0,
    "source": "forecast",
}


@pytest.fixture
def mock_live_round(monkeypatch):
    """Patch ergast_get + fetch_forecast_weather for the fixture round above.

    ergast_get is patched in BOTH f1_predict.ingest (fetch_race_schedule_entry
    calls its own module-level import of it) and f1_predict.live_features
    (used directly there for quali/sprint) — patching only one silently
    leaves the other making a real network call. Returns the AsyncMock
    used for both, so a test can assert on call args or override
    .side_effect for an error scenario.
    """
    import f1_predict.ingest as ingest
    import f1_predict.live_features as live_features

    async def fake_ergast_get(path: str, *args, **kwargs):
        if path == f"{SEASON}/{ROUND_NO}":
            return schedule_response()
        if path == f"{SEASON}/{ROUND_NO}/qualifying":
            return qualifying_response()
        if path == f"{SEASON}/{ROUND_NO}/sprint":
            return EMPTY_RACE_TABLE
        raise AssertionError(f"unexpected ergast_get path in test: {path}")

    ergast_mock = AsyncMock(side_effect=fake_ergast_get)
    weather_mock = AsyncMock(return_value=dict(FORECAST_WEATHER))
    monkeypatch.setattr(ingest, "ergast_get", ergast_mock)
    monkeypatch.setattr(live_features, "ergast_get", ergast_mock)
    monkeypatch.setattr(live_features, "fetch_forecast_weather", weather_mock)
    return ergast_mock, weather_mock

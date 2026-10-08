"""Tests for the live (pre-race) feature path. All network calls mocked
via tests/conftest.py's mock_live_round fixture — see its docstring for
why no synthetic history fixture is needed.
"""

import asyncio

import pandas as pd
import pytest

from f1_predict import ingest, live_features
from tests.conftest import EMPTY_RACE_TABLE, ROUND_NO, SEASON, schedule_response


def test_fetch_live_round_builds_shell_and_quali(mock_live_round):
    round_data = asyncio.run(live_features.fetch_live_round(SEASON, ROUND_NO))

    assert round_data.circuit_id == "marina_bay"
    assert round_data.latitude == pytest.approx(1.2914)
    assert round_data.qualifying is not None
    assert len(round_data.qualifying["QualifyingResults"]) == 2
    assert round_data.sprint is None  # fixture returns an empty sprint table


def test_fetch_live_round_raises_when_round_not_on_calendar(monkeypatch):
    async def fake_ergast_get(path, *args, **kwargs):
        return EMPTY_RACE_TABLE

    # fetch_race_schedule_entry (called first) lives in f1_predict.ingest
    # and uses ITS OWN imported ergast_get — must patch that one too, or
    # this falls through to a real network call.
    monkeypatch.setattr(ingest, "ergast_get", fake_ergast_get)
    monkeypatch.setattr(live_features, "ergast_get", fake_ergast_get)

    with pytest.raises(live_features.QualifyingNotAvailable):
        asyncio.run(live_features.fetch_live_round(2099, 1))


def test_fetch_live_round_raises_when_qualifying_not_published_yet(monkeypatch):
    async def fake_ergast_get(path, *args, **kwargs):
        if path == f"{SEASON}/{ROUND_NO}":
            return schedule_response()
        if path == f"{SEASON}/{ROUND_NO}/qualifying":
            return EMPTY_RACE_TABLE  # round exists, but no quali yet
        raise AssertionError(path)

    monkeypatch.setattr(ingest, "ergast_get", fake_ergast_get)
    monkeypatch.setattr(live_features, "ergast_get", fake_ergast_get)

    with pytest.raises(live_features.QualifyingNotAvailable):
        asyncio.run(live_features.fetch_live_round(SEASON, ROUND_NO))


def test_build_live_stub_rows_shape(mock_live_round):
    round_data = asyncio.run(live_features.fetch_live_round(SEASON, ROUND_NO))
    rows = asyncio.run(live_features.build_live_stub_rows(round_data))

    assert len(rows) == 2
    pole = next(r for r in rows if r["driver_id"] == "max_verstappen")
    assert pole["quali_position"] == 1
    assert pole["grid_position"] == 1  # approximated from quali_position — see live_features docstring
    assert pole["gap_to_pole_s"] == 0.0
    assert pole["finishing_position"] is None  # outcome unknown — this is what's being predicted
    assert pole["weather_source"] == "forecast"
    assert pole["weather_temp_max_c"] == 30.0


def test_build_live_stub_rows_no_sprint_fields_when_no_sprint(mock_live_round):
    round_data = asyncio.run(live_features.fetch_live_round(SEASON, ROUND_NO))
    rows = asyncio.run(live_features.build_live_stub_rows(round_data))

    assert all(r["sprint_position"] is None for r in rows)


def test_build_live_feature_rows_runs_the_real_feature_pipeline(mock_live_round):
    """End-to-end against the real committed race_dataset.parquet as history.

    Confirms the batch feature pipeline (season form, track history,
    racecraft) runs unmodified on live stub rows and produces non-null
    values for a well-known driver with a long history.
    """
    result = asyncio.run(live_features.build_live_feature_rows(SEASON, ROUND_NO))

    assert len(result) == 2
    assert {"driver_points_season_so_far", "driver_racecraft_avg_delta", "era"}.issubset(result.columns)

    ver = result[result["driver_id"] == "max_verstappen"].iloc[0]
    assert ver["driver_points_season_so_far"] > 0  # has raced plenty this season already
    assert ver["era"] == "2026"
    # finishing_position/won/points must stay unknown all the way through —
    # nothing in the pipeline should backfill or guess them.
    assert pd.isna(ver["finishing_position"])

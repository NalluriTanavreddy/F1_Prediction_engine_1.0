"""Tests for the FastAPI service. Network calls mocked via
tests/conftest.py's mock_live_round fixture; models/model.pkl and
data/processed/race_dataset.parquet are the real committed files.
"""

import pytest
from fastapi.testclient import TestClient

from f1_predict.api import app
from tests.conftest import EMPTY_RACE_TABLE, ROUND_NO, SEASON

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_ranked_drivers(mock_live_round) -> None:
    response = client.get(f"/predict/{SEASON}/{ROUND_NO}")
    assert response.status_code == 200

    body = response.json()
    assert body["season"] == SEASON
    assert body["round"] == ROUND_NO
    assert body["race_name"] == "Singapore Grand Prix"
    assert body["circuit_id"] == "marina_bay"

    drivers = body["drivers"]
    assert len(drivers) == 2
    assert all(0.0 <= d["win_probability"] <= 1.0 for d in drivers)
    probability_sum = sum(d["win_probability"] for d in drivers)
    assert probability_sum == pytest.approx(1.0, abs=1e-3)
    assert drivers == sorted(drivers, key=lambda d: d["win_probability"], reverse=True)

    metadata = body["metadata"]
    assert metadata["model_name"] == "LGBMClassifier"
    assert metadata["model_trained_through_season"] == 2025
    assert metadata["weather_source"] == ["forecast"]


def test_predict_404_when_round_does_not_exist(monkeypatch) -> None:
    async def fake_ergast_get(path, *args, **kwargs):
        return EMPTY_RACE_TABLE

    # fetch_race_schedule_entry lives in f1_predict.ingest and uses its own
    # imported ergast_get — patching only live_features's leaves this one
    # hitting the real network (see tests/conftest.py's mock_live_round).
    monkeypatch.setattr("f1_predict.ingest.ergast_get", fake_ergast_get)
    monkeypatch.setattr("f1_predict.live_features.ergast_get", fake_ergast_get)

    response = client.get("/predict/2099/1")
    assert response.status_code == 404
    assert "2099" in response.json()["detail"]

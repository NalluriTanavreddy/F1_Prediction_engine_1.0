"""Load the trained model and produce ranked win-probability predictions.

Predicted probabilities are normalized to sum to 1 within the race — the
model is an independent per-row binary classifier (see evaluate.py's
normalize_per_race for the same reasoning), so its raw predict_proba
output has no reason to already sum to 1 across a race's field.
"""

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import pandas as pd

from f1_predict.clients.http import close_client
from f1_predict.live_features import QualifyingNotAvailable, build_live_feature_rows
from f1_predict.train import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

MODEL_PATH = Path("models/model.pkl")
MODEL_TRAINED_THROUGH_SEASON = 2025  # models/model.pkl was trained on 2022-2025; see README


def load_model_bundle(model_path: Path = MODEL_PATH) -> dict:
    with open(model_path, "rb") as f:
        return pickle.load(f)


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Same column selection/dtype casting as train.prepare_X_y, minus the
    target — a live feature row has no `won` column to cast (the race
    hasn't happened). Deliberately not importing prepare_X_y itself: this
    session must not touch train.py/the trained model at all, and X-prep
    needs to work on target-less rows that prepare_X_y was never meant to.
    """
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    return X


def rank_predictions(bundle: dict, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Score feature_df, normalize per-race, return sorted by win probability descending."""
    X = prepare_X(feature_df)
    raw_proba = bundle["model"].predict_proba(X)[:, 1]
    result = feature_df.copy()
    result["win_probability_raw"] = raw_proba
    result["win_probability"] = raw_proba / raw_proba.sum()
    return result.sort_values("win_probability", ascending=False).reset_index(drop=True)


async def predict_race(season: int, round_no: int, *, bundle: dict | None = None) -> dict:
    """Full live prediction for one round: ranked drivers + a metadata block.

    Raises live_features.QualifyingNotAvailable if the round doesn't
    exist yet or hasn't had qualifying published.
    """
    feature_df = await build_live_feature_rows(season, round_no)
    bundle = bundle if bundle is not None else load_model_bundle()
    ranked = rank_predictions(bundle, feature_df)

    weather_sources = ranked["weather_source"].unique().tolist()
    drivers = [
        {
            "driver_id": row["driver_id"],
            "driver_code": row["driver_code"],
            "driver_name": row["driver_name"],
            "constructor_id": row["constructor_id"],
            "grid_position": None if pd.isna(row["grid_position"]) else int(row["grid_position"]),
            "win_probability": round(float(row["win_probability"]), 4),
        }
        for _, row in ranked.iterrows()
    ]
    return {
        "season": season,
        "round": round_no,
        "race_name": ranked["race_name"].iloc[0],
        "circuit_id": ranked["circuit_id"].iloc[0],
        "drivers": drivers,
        "metadata": {
            "model_name": type(bundle["model"]).__name__,
            "model_trained_through_season": MODEL_TRAINED_THROUGH_SEASON,
            "prediction_generated_at": pd.Timestamp.now("UTC").isoformat(),
            # One weather lookup per race (see live_features.py), so this
            # is always a single value in practice; kept as a list so a
            # caller can tell if that ever stopped being true.
            "weather_source": weather_sources,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict win probabilities for an F1 race from live data.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_no")
    args = parser.parse_args()

    async def _run() -> dict:
        try:
            return await predict_race(args.season, args.round_no)
        finally:
            await close_client()

    try:
        result = asyncio.run(_run())
    except QualifyingNotAvailable as exc:
        print(f"Cannot predict: {exc}")
        raise SystemExit(1)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

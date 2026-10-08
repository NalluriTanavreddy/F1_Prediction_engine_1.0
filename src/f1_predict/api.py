"""FastAPI service for live race-winner predictions.

GET /health
GET /predict/{season}/{round} — ranked drivers with win probability, grid
position, and a metadata block (model name, what season the model was
trained through, when the prediction was generated, and whether the
underlying weather is a forecast or — if run against a completed round,
e.g. for the skew check — historical actuals).

Run with:
    uv run uvicorn f1_predict.api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from f1_predict.live_features import QualifyingNotAvailable
from f1_predict.predict import predict_race

app = FastAPI(
    title="F1 Race Winner Prediction API",
    description=(
        "Predicts race-winner probabilities from post-qualifying data. "
        "LightGBM, ties the pole-position baseline on top-1 accuracy and "
        "improves log loss — see the README's Evaluation/Validation "
        "sections before reading too much into any single prediction; "
        "the holdout this was checked against is only 16 races."
    ),
)


class DriverPrediction(BaseModel):
    driver_id: str
    driver_code: str | None
    driver_name: str
    constructor_id: str
    grid_position: int | None
    win_probability: float


class PredictionMetadata(BaseModel):
    model_name: str
    model_trained_through_season: int
    prediction_generated_at: str
    weather_source: list[str]


class PredictionResponse(BaseModel):
    season: int
    round: int
    race_name: str
    circuit_id: str
    drivers: list[DriverPrediction]
    metadata: PredictionMetadata


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/predict/{season}/{round}", response_model=PredictionResponse)
async def predict(season: int, round: int) -> dict:
    try:
        return await predict_race(season, round)
    except QualifyingNotAvailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

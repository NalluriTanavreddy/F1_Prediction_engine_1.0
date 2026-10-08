"""Live (pre-race) feature construction for a single (season, round).

Reuses the EXACT same feature-engineering code as the batch pipeline
(f1_predict.features.build_features) — this module's only job is to build
"stub" rows for a race whose result we don't have (or are deliberately
ignoring, for the skew check), append them to history, and let the same
shift(1)-based transforms compute pre-race feature values for them.

Reused unchanged from f1_predict.build_dataset: _quali_rows, _sprint_rows,
era_for_season — all pure functions of RoundData that never read
round_data.results, so they work identically whether or not the race has
happened.

Reused unchanged from f1_predict.features: build_features and everything
it calls. Every rolling/cumulative feature there is built as
shift(1).<agg>(...) (or <cumagg>().shift(1)) computed per group in
chronological order — a stub row's own (unknown) finishing_position/
points/won is never read to compute its OWN feature value, and since the
stub row is chronologically last in its group (history is everything
strictly before it), nothing else depends on it either. Season form,
track history, and racecraft tendency all work with zero code changes.
This is verified empirically, not just argued: see
f1_predict.skew_check, which runs this path on completed 2026 rounds and
diffs the result against the offline model_dataset.parquet.

The one feature that's genuinely approximated, not reused: grid_position.
The trained model's grid_position is the real post-penalty starting grid,
published shortly before the race — it doesn't exist yet right after
qualifying. The best available estimate at prediction time is
quali_position (no penalty applied). This is the one column the skew
check expects to sometimes differ from the offline dataset, whenever a
real grid penalty was issued for that race.
"""

from pathlib import Path

import pandas as pd

from f1_predict.build_dataset import _quali_rows, _sprint_rows, era_for_season
from f1_predict.clients.ergast import ergast_get
from f1_predict.clients.open_meteo import fetch_forecast_weather
from f1_predict.features import build_features
from f1_predict.ingest import RoundData, fetch_race_schedule_entry, round_data_shell

RACE_DATASET_PATH = Path("data/processed/race_dataset.parquet")


class QualifyingNotAvailable(Exception):
    """Raised when a round doesn't exist yet, or has no published qualifying session."""


async def fetch_live_round(season: int, round_no: int) -> RoundData:
    """Fetch a round's schedule + qualifying + sprint (never results) for a live prediction."""
    race = await fetch_race_schedule_entry(season, round_no)
    if race is None:
        raise QualifyingNotAvailable(f"No {season} round {round_no} on the calendar")

    round_data = round_data_shell(season, race)

    quali_data = await ergast_get(f"{season}/{round_no}/qualifying")
    quali_races = quali_data["MRData"]["RaceTable"].get("Races", [])
    if not quali_races or not quali_races[0].get("QualifyingResults"):
        raise QualifyingNotAvailable(f"Qualifying for {season} round {round_no} isn't available yet")
    round_data.qualifying = quali_races[0]

    # Deliberately NOT cache_empty=True here, unlike ingest.py's batch
    # path: we don't know this round is "complete" the way a finished
    # results.json tells ingest.py, so "no sprint yet" must stay
    # re-checkable right up until the main race, not cached as permanent.
    sprint_data = await ergast_get(f"{season}/{round_no}/sprint")
    sprint_races = sprint_data["MRData"]["RaceTable"].get("Races", [])
    round_data.sprint = sprint_races[0] if sprint_races else None

    return round_data


def _driver_constructor_identity(entry: dict) -> dict:
    driver = entry["Driver"]
    constructor = entry["Constructor"]
    return {
        "driver_id": driver["driverId"],
        "driver_code": driver.get("code"),
        "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
        "constructor_id": constructor["constructorId"],
        "constructor_name": constructor.get("name"),
    }


async def build_live_stub_rows(round_data: RoundData, *, force_refresh_weather: bool = True) -> list[dict]:
    """One row per driver entered in qualifying, with every pre-race-known field filled in.

    finishing_position/points/won/status are left unknown — that's what
    the model is being asked to predict. dnf is an unused placeholder
    (features.py never reads it).
    """
    quali_entries = round_data.qualifying.get("QualifyingResults", [])
    quali_fields_by_driver = _quali_rows(round_data)
    sprint_fields_by_driver = _sprint_rows(round_data)

    weather = await fetch_forecast_weather(
        round_data.latitude, round_data.longitude, round_data.date, force_refresh=force_refresh_weather
    )

    rows = []
    for entry in quali_entries:
        identity = _driver_constructor_identity(entry)
        quali_fields = quali_fields_by_driver[identity["driver_id"]]
        sprint_fields = sprint_fields_by_driver.get(identity["driver_id"], {"sprint_position": None, "sprint_points": None})

        rows.append(
            {
                "season": round_data.season,
                "round": round_data.round,
                "era": era_for_season(round_data.season),
                "race_name": round_data.race_name,
                "circuit_id": round_data.circuit_id,
                "circuit_name": round_data.circuit_name,
                "country": round_data.country,
                "latitude": round_data.latitude,
                "longitude": round_data.longitude,
                "date": round_data.date,
                **identity,
                **quali_fields,
                "grid_position": quali_fields["quali_position"],  # see module docstring
                "finishing_position": None,
                "won": float("nan"),
                "points": float("nan"),
                "status": None,
                "dnf": False,
                **sprint_fields,
                "weather_temp_max_c": weather["temp_max_c"],
                "weather_precip_mm": weather["precip_mm"],
                "weather_wind_speed_max_kph": weather["wind_speed_max_kph"],
                "weather_rain_probability_pct": weather["rain_probability_pct"],
                "weather_source": weather["source"],
                "missing_qualifying_data": False,
            }
        )
    return rows


def load_history_before(target_date: str) -> pd.DataFrame:
    """Every race_dataset.parquet row strictly before target_date.

    Excludes the target round itself, even if it's already in the dataset
    (e.g. the skew check runs this on completed rounds) — its real result
    must never leak into its own feature computation.
    """
    df = pd.read_parquet(RACE_DATASET_PATH)
    df["era"] = df["era"].astype(str)  # re-categorized fresh by build_features on the combined frame
    cutoff = pd.Timestamp(target_date)
    return df[df["date"] < cutoff].reset_index(drop=True)


NULLABLE_INT_COLUMNS = ["quali_position", "grid_position", "finishing_position", "sprint_position"]
FLOAT_COLUMNS = [
    "q1_time_s",
    "q2_time_s",
    "q3_time_s",
    "best_quali_time_s",
    "gap_to_pole_s",
    "won",
    "points",
    "sprint_points",
    "weather_temp_max_c",
    "weather_precip_mm",
    "weather_wind_speed_max_kph",
    "weather_rain_probability_pct",
]


def _cast_stub_dtypes(stub_df: pd.DataFrame) -> pd.DataFrame:
    """Match race_dataset.parquet's dtypes exactly.

    Without this, columns the stub fills with None (finishing_position,
    points, ...) come back as pandas object dtype, and concatenating an
    object column with history's Int64/float64 column produces an object
    column overall — which then breaks every numeric transform in
    build_features (rolling/cumsum/etc. can't aggregate object dtype).
    """
    stub_df = stub_df.copy()
    for col in NULLABLE_INT_COLUMNS:
        stub_df[col] = stub_df[col].astype("Int64")
    for col in FLOAT_COLUMNS:
        stub_df[col] = stub_df[col].astype("float64")
    stub_df["date"] = pd.to_datetime(stub_df["date"])
    return stub_df


async def build_live_feature_rows(season: int, round_no: int, *, force_refresh_weather: bool = True) -> pd.DataFrame:
    """Fetch the live round, build stub rows, run the batch feature pipeline, return this round's rows."""
    round_data = await fetch_live_round(season, round_no)
    stub_rows = await build_live_stub_rows(round_data, force_refresh_weather=force_refresh_weather)
    stub_df = _cast_stub_dtypes(pd.DataFrame(stub_rows))

    history = load_history_before(round_data.date)
    combined = pd.concat([history, stub_df], ignore_index=True)
    # Same sort key as features.load_base_dataset — build_features's
    # groupby/shift transforms assume the frame already arrives in
    # chronological order; it doesn't re-sort internally.
    combined = combined.sort_values(["date", "round"], kind="stable").reset_index(drop=True)

    featured = build_features(combined)
    target = featured[(featured["season"] == season) & (featured["round"] == round_no)].reset_index(drop=True)
    return target

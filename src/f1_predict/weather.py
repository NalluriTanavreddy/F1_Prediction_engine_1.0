"""Add race-day weather to race_dataset.parquet.

One weather lookup per (season, round) — not per driver row — since every
driver in a race experiences the same race-day weather at the same circuit.
Reads race_dataset.parquet, adds weather_* columns, and writes it back in
place. Run this after build_dataset.py and before features.py.

Historical races (the only kind currently in the dataset) use Open-Meteo's
archive API: actual observed race-day conditions. This is a known,
accepted tradeoff — see README's "Weather" section for the train/serve
skew this implies, including for the 2026 test-set evaluation. Any race
dated after today would use the forecast API instead; the two paths are
kept in separate functions in clients/open_meteo.py specifically so this
dispatch can never accidentally serve hindsight weather for a live
prediction.
"""

import asyncio
import datetime
from pathlib import Path

import pandas as pd

from f1_predict.clients.http import close_client
from f1_predict.clients.open_meteo import fetch_forecast_weather, fetch_historical_weather

DATASET_PATH = Path("data/processed/race_dataset.parquet")

WEATHER_COLUMNS = [
    "weather_temp_max_c",
    "weather_precip_mm",
    "weather_wind_speed_max_kph",
    "weather_rain_probability_pct",
    "weather_source",
]


async def fetch_race_weather(latitude: float, longitude: float, date: str, today: datetime.date) -> dict:
    """Dispatch to the historical or forecast path based on whether `date` has passed."""
    race_date = datetime.date.fromisoformat(date)
    if race_date <= today:
        return await fetch_historical_weather(latitude, longitude, date)
    return await fetch_forecast_weather(latitude, longitude, date)


async def fetch_all_race_weather(races: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, int, str]]]:
    """races: one row per (season, round) with latitude/longitude/date.

    Returns a (season, round) -> weather_* frame, plus a list of
    (season, round, error) for races whose weather couldn't be fetched
    (e.g. no circuit coordinates).
    """
    today = datetime.date.today()
    rows = []
    failures = []

    for _, race in races.iterrows():
        season, round_no, date = int(race["season"]), int(race["round"]), race["date"]
        lat, lon = race["latitude"], race["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            failures.append((season, round_no, "no circuit coordinates"))
            continue
        try:
            weather = await fetch_race_weather(float(lat), float(lon), date, today)
        except Exception as exc:
            failures.append((season, round_no, str(exc)))
            continue
        rows.append(
            {
                "season": season,
                "round": round_no,
                "weather_temp_max_c": weather["temp_max_c"],
                "weather_precip_mm": weather["precip_mm"],
                "weather_wind_speed_max_kph": weather["wind_speed_max_kph"],
                "weather_rain_probability_pct": weather["rain_probability_pct"],
                "weather_source": weather["source"],
            }
        )

    columns = ["season", "round"] + WEATHER_COLUMNS
    return pd.DataFrame(rows, columns=columns), failures


def add_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    existing = [c for c in WEATHER_COLUMNS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df.merge(weather, on=["season", "round"], how="left")


async def _main() -> None:
    df = pd.read_parquet(DATASET_PATH)
    races = df.drop_duplicates(subset=["season", "round"])[["season", "round", "date", "latitude", "longitude"]]
    races = races.assign(date=races["date"].dt.strftime("%Y-%m-%d"))

    try:
        weather, failures = await fetch_all_race_weather(races)
    finally:
        await close_client()

    df = add_weather(df, weather)
    df.to_parquet(DATASET_PATH, index=False)

    print(f"Fetched weather for {len(weather)} of {len(races)} races")
    source_counts = weather["weather_source"].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    if failures:
        print(f"\nFailed to fetch weather for {len(failures)} races:")
        for season, round_no, error in failures:
            print(f"  {season} round {round_no}: {error}")
    else:
        print("\nFailures: none")
    print(f"\nWrote weather columns to {DATASET_PATH}")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

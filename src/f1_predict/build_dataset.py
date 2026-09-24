"""Reshape ingested season/round data into one row per (driver, race).

Runs the ingestion pipeline (f1_predict.ingest), reshapes the raw Jolpica
qualifying + race results into a flat per-driver-per-race table, validates
it, prints a summary, and writes the result to
data/processed/race_dataset.parquet.
"""

import re
from pathlib import Path

import pandas as pd

from f1_predict.ingest import RoundData, run

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "race_dataset.parquet"

_LAP_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")


def parse_lap_time(raw: str | None) -> float | None:
    """Parse a lap/quali time like '1:29.708' or '29.708' into seconds."""
    if not raw:
        return None
    match = _LAP_TIME_RE.match(raw.strip())
    if not match:
        return None
    minutes, seconds = match.groups()
    total = float(seconds)
    if minutes:
        total += int(minutes) * 60
    return total


def era_for_season(season: int) -> str:
    return "2026" if season >= 2026 else "2022-2025"


def _quali_rows(round_data: RoundData) -> dict[str, dict]:
    """driver_id -> qualifying fields for one round."""
    rows: dict[str, dict] = {}
    if round_data.qualifying is None:
        return rows

    entries = round_data.qualifying.get("QualifyingResults", [])
    best_times: dict[str, float | None] = {}
    for entry in entries:
        driver_id = entry["Driver"]["driverId"]
        q1 = parse_lap_time(entry.get("Q1"))
        q2 = parse_lap_time(entry.get("Q2"))
        q3 = parse_lap_time(entry.get("Q3"))
        best = min((t for t in (q1, q2, q3) if t is not None), default=None)
        best_times[driver_id] = best
        rows[driver_id] = {
            "quali_position": int(entry["position"]) if entry.get("position") else None,
            "q1_time_s": q1,
            "q2_time_s": q2,
            "q3_time_s": q3,
            "best_quali_time_s": best,
        }

    pole_time = None
    for driver_id, row in rows.items():
        if row["quali_position"] == 1:
            pole_time = row["best_quali_time_s"]
            break
    if pole_time is None:
        pole_time = min((t for t in best_times.values() if t is not None), default=None)

    for row in rows.values():
        best = row["best_quali_time_s"]
        row["gap_to_pole_s"] = (best - pole_time) if (best is not None and pole_time is not None) else None

    return rows


def _results_rows(round_data: RoundData) -> list[dict]:
    """One dict per driver who took part in the race."""
    if round_data.results is None:
        return []

    rows = []
    for entry in round_data.results.get("Results", []):
        driver = entry["Driver"]
        constructor = entry["Constructor"]
        position_text = entry.get("positionText", "")
        dnf = not position_text.isdigit()
        finishing_position = None if dnf else int(position_text)
        grid_raw = entry.get("grid")
        grid_position = int(grid_raw) if grid_raw is not None and grid_raw.isdigit() else None

        rows.append(
            {
                "driver_id": driver["driverId"],
                "driver_code": driver.get("code"),
                "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                "constructor_id": constructor["constructorId"],
                "constructor_name": constructor.get("name"),
                "grid_position": grid_position,
                "finishing_position": finishing_position,
                "points": float(entry.get("points", 0) or 0),
                "status": entry.get("status"),
                "dnf": dnf,
            }
        )
    return rows


def _sprint_rows(round_data: RoundData) -> dict[str, dict]:
    """driver_id -> sprint fields for one round. Empty dict if no sprint that weekend.

    Sprint runs the day before the main race, so its result is known ahead
    of time — a legitimate pre-race feature, not a leakage risk.
    """
    rows: dict[str, dict] = {}
    if round_data.sprint is None:
        return rows

    for entry in round_data.sprint.get("SprintResults", []):
        driver_id = entry["Driver"]["driverId"]
        position_text = entry.get("positionText", "")
        sprint_dnf = not position_text.isdigit()
        rows[driver_id] = {
            "sprint_position": None if sprint_dnf else int(position_text),
            "sprint_points": float(entry.get("points", 0) or 0),
        }
    return rows


def build_race_rows(round_data: RoundData) -> list[dict]:
    """All per-driver rows for one round, quali + results + sprint merged."""
    quali_by_driver = _quali_rows(round_data)
    missing_qualifying = round_data.qualifying is None or not quali_by_driver
    results = _results_rows(round_data)
    sprint_by_driver = _sprint_rows(round_data)

    empty_quali_fields = {
        "quali_position": None,
        "q1_time_s": None,
        "q2_time_s": None,
        "q3_time_s": None,
        "best_quali_time_s": None,
        "gap_to_pole_s": None,
    }
    empty_sprint_fields = {"sprint_position": None, "sprint_points": None}

    rows = []
    for result_row in results:
        quali_fields = quali_by_driver.get(result_row["driver_id"], empty_quali_fields)
        sprint_fields = sprint_by_driver.get(result_row["driver_id"], empty_sprint_fields)
        row = {
            "season": round_data.season,
            "round": round_data.round,
            "era": era_for_season(round_data.season),
            "race_name": round_data.race_name,
            "circuit_id": round_data.circuit_id,
            "circuit_name": round_data.circuit_name,
            "country": round_data.country,
            "date": round_data.date,
            **result_row,
            **quali_fields,
            **sprint_fields,
            "missing_qualifying_data": missing_qualifying,
        }
        row["won"] = int(row["finishing_position"] == 1)
        rows.append(row)
    return rows


COLUMN_ORDER = [
    "season",
    "round",
    "era",
    "race_name",
    "circuit_id",
    "circuit_name",
    "country",
    "date",
    "driver_id",
    "driver_code",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "quali_position",
    "grid_position",
    "q1_time_s",
    "q2_time_s",
    "q3_time_s",
    "best_quali_time_s",
    "gap_to_pole_s",
    "finishing_position",
    "won",
    "points",
    "status",
    "dnf",
    "sprint_position",
    "sprint_points",
    "missing_qualifying_data",
]

NULLABLE_INT_COLUMNS = ["quali_position", "grid_position", "finishing_position", "sprint_position"]


def build_dataframe(rounds: list[RoundData]) -> pd.DataFrame:
    all_rows = [row for round_data in rounds for row in build_race_rows(round_data)]
    df = pd.DataFrame(all_rows, columns=COLUMN_ORDER)
    for col in NULLABLE_INT_COLUMNS:
        df[col] = df[col].astype("Int64")
    df["era"] = df["era"].astype("category")
    df["date"] = pd.to_datetime(df["date"])
    return df


def print_summary(df: pd.DataFrame, rounds: list[RoundData], failures: list[tuple[int, int, str]]) -> None:
    total_races = df.drop_duplicates(subset=["season", "round"]).shape[0]

    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total rows:   {len(df)}")
    print(f"Total races:  {total_races}")

    print("\nRows per season:")
    for season, count in df.groupby("season").size().items():
        races = df[df["season"] == season].drop_duplicates(subset=["round"]).shape[0]
        print(f"  {season}: {count} rows across {races} races")

    print("\nRows per era:")
    for era, count in df.groupby("era", observed=True).size().items():
        races = df[df["era"] == era].drop_duplicates(subset=["season", "round"]).shape[0]
        print(f"  {era}: {count} rows across {races} races")

    dnf_count = int(df["dnf"].sum())
    print(f"\nDNF rows: {dnf_count} ({dnf_count / len(df):.1%} of all rows)")

    sprint_races = df.loc[df["sprint_position"].notna() | df["sprint_points"].notna(), ["season", "round"]]
    sprint_races = sprint_races.drop_duplicates()
    print(f"\nSprint weekends: {len(sprint_races)} of {total_races} races")

    missing_quali = df.loc[df["missing_qualifying_data"], ["season", "round", "race_name"]].drop_duplicates()
    if len(missing_quali):
        print(f"\nRaces with no qualifying data ({len(missing_quali)}):")
        for _, r in missing_quali.iterrows():
            print(f"  {r['season']} round {r['round']}: {r['race_name']}")
    else:
        print("\nRaces with no qualifying data: none")

    if failures:
        print(f"\nRounds that failed to fetch ({len(failures)}):")
        for season, round_no, error in failures:
            print(f"  {season} round {round_no}: {error}")
    else:
        print("\nRounds that failed to fetch: none")
    print("=" * 60)


def main() -> None:
    ingest_result = run()
    df = build_dataframe(ingest_result.rounds)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}\n")

    print_summary(df, ingest_result.rounds, ingest_result.failures)


if __name__ == "__main__":
    main()

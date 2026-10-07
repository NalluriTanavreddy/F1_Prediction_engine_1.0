"""Feature engineering on the per-(driver, race) base dataset.

Reads data/processed/race_dataset.parquet and adds:
- season-form features (driver + constructor): points so far, rolling
  avg finish / win rate / podium rate over the last N races
- track-history features: driver/constructor historical avg finish at
  this circuit, across all prior seasons
- racecraft tendency: a driver's career-long avg grid-to-finish delta
- native pandas `category` dtype for driver_id/constructor_id/circuit_id/era
- a chronological train/test split (train: 2022-2025, test: 2026)

Writes data/processed/model_dataset.parquet.

Every rolling/cumulative feature is built as group.shift(1).<agg>() (or
group.<cumagg>().shift(1)) — the value for row i is always computed from
rows strictly before it in the group's chronological order, never row i
itself. See validate_no_leakage() for the structural checks this implies.
"""

from pathlib import Path

import pandas as pd

BASE_DATASET_PATH = Path("data/processed/race_dataset.parquet")
OUTPUT_PATH = Path("data/processed/model_dataset.parquet")

ROLLING_WINDOWS = (3, 5)
CATEGORICAL_COLUMNS = ["driver_id", "constructor_id", "circuit_id", "era"]
TEST_SEASON = 2026


def load_base_dataset() -> pd.DataFrame:
    df = pd.read_parquet(BASE_DATASET_PATH)
    return df.sort_values(["date", "round"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Season form
# ---------------------------------------------------------------------------
#
# Points-so-far-this-season resets every season by definition (a driver
# legitimately has 0 championship points before round 1 — that's real
# information, not a missing value).
#
# Rolling avg-finish / win-rate / podium-rate are NOT scoped to "this
# season" — they're grouped by (entity, era) instead, so round 1 of a new
# season still carries over the driver's form from the end of the previous
# season *as long as it's the same regulation era*. At the 2025->2026 era
# boundary that continuity is deliberately broken: a driver's 2022-2025
# form isn't assumed predictive of 2026 pace under all-new regulations, so
# those rows start with null form (until enough 2026 races accumulate) and
# gradient-boosted trees split on that null natively.


def _add_rolling_form(df: pd.DataFrame, group_keys: list[str], prefix: str, position_col: str, points_col: str, win_col: str) -> None:
    """Add <prefix>_avg_finish_lastN, _win_rate_lastN, _podium_rate_lastN in place."""
    podium_col = f"_{prefix}_podium_tmp"
    df[podium_col] = (df[position_col] <= 3).fillna(False).astype(int)

    grouped = df.groupby(group_keys, observed=True)
    for n in ROLLING_WINDOWS:
        df[f"{prefix}_avg_finish_last{n}"] = grouped[position_col].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"{prefix}_win_rate_last{n}"] = grouped[win_col].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"{prefix}_podium_rate_last{n}"] = grouped[podium_col].transform(
            lambda s, n=n: s.shift(1).rolling(n, min_periods=1).mean()
        )

    df.drop(columns=[podium_col], inplace=True)


def add_driver_season_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driver_points_season_so_far"] = df.groupby(["driver_id", "season"], observed=True)["points"].transform(
        lambda s: s.cumsum().shift(1).fillna(0)
    )
    _add_rolling_form(
        df,
        group_keys=["driver_id", "era"],
        prefix="driver",
        position_col="finishing_position",
        points_col="points",
        win_col="won",
    )
    return df


def add_constructor_season_form(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per (constructor, race), compute rolling form there, merge back.

    Both cars on a team share the same constructor-form value for a given
    race — that value only depends on races strictly before this one, so
    it's identical and leak-free for both drivers.
    """
    df = df.copy()

    race_agg = (
        df.groupby(["constructor_id", "season", "round", "era", "date"], observed=True)
        .agg(
            constructor_points_race=("points", "sum"),
            constructor_avg_finish_race=("finishing_position", "mean"),
            constructor_win=("won", "max"),
        )
        .reset_index()
    )
    race_agg = race_agg.sort_values(["date", "round"], kind="stable").reset_index(drop=True)

    race_agg["constructor_points_season_so_far"] = race_agg.groupby(
        ["constructor_id", "season"], observed=True
    )["constructor_points_race"].transform(lambda s: s.cumsum().shift(1).fillna(0))

    _add_rolling_form(
        race_agg,
        group_keys=["constructor_id", "era"],
        prefix="constructor",
        position_col="constructor_avg_finish_race",
        points_col="constructor_points_race",
        win_col="constructor_win",
    )

    # Only merge the lagged (shift(1)-based) columns back — race_agg also
    # holds unshifted per-race aggregates (constructor_points_race,
    # constructor_avg_finish_race, constructor_win) used purely as rolling
    # inputs above. Those describe *this* race's own outcome for both
    # teammates and must never end up in the model table as a "feature".
    feature_cols = ["constructor_points_season_so_far"] + [
        f"constructor_{stat}_last{n}" for stat in ("avg_finish", "win_rate", "podium_rate") for n in ROLLING_WINDOWS
    ]
    merge_keys = ["constructor_id", "season", "round"]
    df = df.merge(race_agg[merge_keys + feature_cols], on=merge_keys, how="left")
    return df


# ---------------------------------------------------------------------------
# Track history — uses ALL prior seasons (not era-scoped: circuits and
# their characteristics don't change with the power unit/aero regs, so
# there's no reason to discard 2022-2025 track history for a 2026 row).
# ---------------------------------------------------------------------------


def add_track_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driver_circuit_avg_finish"] = df.groupby(["driver_id", "circuit_id"], observed=True)[
        "finishing_position"
    ].transform(lambda s: s.shift(1).expanding().mean())

    constructor_circuit_race = (
        df.groupby(["constructor_id", "circuit_id", "season", "round", "date"], observed=True)["finishing_position"]
        .mean()
        .reset_index()
        .sort_values(["date", "round"], kind="stable")
    )
    constructor_circuit_race["constructor_circuit_avg_finish"] = constructor_circuit_race.groupby(
        ["constructor_id", "circuit_id"], observed=True
    )["finishing_position"].transform(lambda s: s.shift(1).expanding().mean())

    merge_keys = ["constructor_id", "circuit_id", "season", "round"]
    df = df.merge(constructor_circuit_race[merge_keys + ["constructor_circuit_avg_finish"]], on=merge_keys, how="left")
    return df


# ---------------------------------------------------------------------------
# Racecraft tendency — career-long (not era-scoped: overtaking/racecraft
# skill is more a driver trait than a car/regulation trait), no track-type
# grouping (26 circuits over 106 races is too little to slice further —
# user decision).
# ---------------------------------------------------------------------------


def add_racecraft_tendency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grid_finish_delta = df["grid_position"] - df["finishing_position"]
    df["driver_racecraft_avg_delta"] = (
        grid_finish_delta.groupby(df["driver_id"], observed=True)
        .transform(lambda s: s.shift(1).expanding().mean())
    )
    return df


# ---------------------------------------------------------------------------
# Categorical dtype + split
# ---------------------------------------------------------------------------


def add_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    return df


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["season"].apply(lambda s: "test" if s >= TEST_SEASON else "train")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_driver_season_form(df)
    df = add_constructor_season_form(df)
    df = add_track_history(df)
    df = add_racecraft_tendency(df)
    df = add_categorical_dtypes(df)
    df = add_split(df)
    return df


FEATURE_COLUMNS = [
    "driver_points_season_so_far",
    "driver_avg_finish_last3",
    "driver_win_rate_last3",
    "driver_podium_rate_last3",
    "driver_avg_finish_last5",
    "driver_win_rate_last5",
    "driver_podium_rate_last5",
    "constructor_points_season_so_far",
    "constructor_avg_finish_last3",
    "constructor_win_rate_last3",
    "constructor_podium_rate_last3",
    "constructor_avg_finish_last5",
    "constructor_win_rate_last5",
    "constructor_podium_rate_last5",
    "driver_circuit_avg_finish",
    "constructor_circuit_avg_finish",
    "driver_racecraft_avg_delta",
    "sprint_position",
    "sprint_points",
    "weather_temp_max_c",
    "weather_precip_mm",
    "weather_wind_speed_max_kph",
    "weather_rain_probability_pct",
]


def print_completeness(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("FEATURE COMPLETENESS (% non-null)")
    print("=" * 60)
    for col in FEATURE_COLUMNS:
        pct = 100 * df[col].notna().mean()
        print(f"  {col:<36} {pct:6.1f}%")
    print()
    print("Feature completeness by split:")
    for split_name, group in df.groupby("split"):
        print(f"  {split_name}:")
        for col in FEATURE_COLUMNS:
            pct = 100 * group[col].notna().mean()
            print(f"    {col:<34} {pct:6.1f}%")
    print("=" * 60)


def validate_no_leakage(df: pd.DataFrame) -> None:
    """Structural leakage checks + a printed spot check.

    These follow directly from how every feature is built (shift(1) before
    any rolling/cumulative/expanding aggregation), but assert them so a
    future refactor that breaks the invariant fails loudly instead of
    silently leaking.
    """
    print("=" * 60)
    print("LEAKAGE CHECKS")
    print("=" * 60)

    # 1. A driver's first-ever race in an era must have null rolling form —
    #    there is nothing before it to compute from.
    first_era_race = df.sort_values(["date", "round"]).groupby(["driver_id", "era"], observed=True).head(1)
    bad = first_era_race[first_era_race["driver_avg_finish_last3"].notna()]
    assert bad.empty, f"driver's first race in an era has non-null rolling form:\n{bad}"
    print("OK: every driver's first race in an era has null rolling-form features")

    # 2. Points-so-far for a driver's first race of a season must be exactly 0.
    first_season_race = df.sort_values(["date", "round"]).groupby(["driver_id", "season"], observed=True).head(1)
    bad = first_season_race[first_season_race["driver_points_season_so_far"] != 0]
    assert bad.empty, f"driver's first race of a season has nonzero points_season_so_far:\n{bad}"
    print("OK: every driver's first race of a season has driver_points_season_so_far == 0")

    # 3. A driver/circuit combo's first-ever visit must have null track history.
    first_circuit_visit = (
        df.sort_values(["date", "round"]).groupby(["driver_id", "circuit_id"], observed=True).head(1)
    )
    bad = first_circuit_visit[first_circuit_visit["driver_circuit_avg_finish"].notna()]
    assert bad.empty, f"driver's first visit to a circuit has non-null track history:\n{bad}"
    print("OK: every driver's first visit to a circuit has null driver_circuit_avg_finish")

    # 4. Spot check: manually recompute driver_points_season_so_far for one
    #    driver/season from only strictly-prior rows and compare.
    sample_driver, sample_season = df.iloc[len(df) // 2][["driver_id", "season"]]
    season_races = df[(df["driver_id"] == sample_driver) & (df["season"] == sample_season)].sort_values("round")
    print(f"\nSpot check — {sample_driver}, {sample_season} season:")
    running_points = 0.0
    for _, row in season_races.iterrows():
        expected = running_points
        actual = row["driver_points_season_so_far"]
        status = "OK" if expected == actual else "MISMATCH"
        print(f"  round {row['round']:>2}: points_season_so_far={actual:>6} expected={expected:>6}  [{status}]")
        assert expected == actual, f"leakage in driver_points_season_so_far at round {row['round']}"
        running_points += row["points"]

    print("=" * 60)


def print_split_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("TRAIN/TEST SPLIT")
    print("=" * 60)
    for split_name, group in df.groupby("split"):
        races = group.drop_duplicates(subset=["season", "round"]).shape[0]
        seasons = sorted(group["season"].unique().tolist())
        print(f"  {split_name}: {len(group)} rows, {races} races, seasons {seasons}")
    print("=" * 60)


def main() -> None:
    df = load_base_dataset()
    df = build_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}\n")

    print_completeness(df)
    print()
    validate_no_leakage(df)
    print()
    print_split_summary(df)


if __name__ == "__main__":
    main()

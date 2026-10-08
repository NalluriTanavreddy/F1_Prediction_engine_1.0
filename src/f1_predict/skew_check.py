"""Skew check: compare the live feature path against the offline dataset.

Runs live_features.build_live_feature_rows on a COMPLETED round (ignoring
its real result — see load_history_before), and diffs the result against
the already-built data/processed/model_dataset.parquet for that same
round: feature-by-feature, and the resulting predicted probabilities.

Expected, not bugs:
- weather_* columns differ: offline uses the historical archive (actual
  conditions); live uses the forecast API. Only possible to compare them
  at all within Open-Meteo's forecast window (roughly the last ~3 months
  — see clients/open_meteo.py), which is why this only runs against
  recent rounds, not arbitrary history.
- grid_position can differ: the live path estimates it as quali_position
  (no penalty applied yet — see live_features.py); offline uses the real
  post-penalty grid from race results. Matches whenever no penalty was
  issued, differs otherwise.

Any OTHER column differing (season form, track history, racecraft,
quali times, gap to pole, sprint result) would indicate a real bug in
the live reconstruction, since those don't depend on a forecast-vs-actual
distinction at all.
"""

import asyncio

import pandas as pd

from f1_predict.clients.http import close_client
from f1_predict.live_features import build_live_feature_rows
from f1_predict.predict import load_model_bundle, rank_predictions
from f1_predict.train import FEATURE_COLUMNS

MODEL_DATASET_PATH = "data/processed/model_dataset.parquet"

EXPECTED_TO_DIFFER = {
    "weather_temp_max_c",
    "weather_precip_mm",
    "weather_wind_speed_max_kph",
    "weather_rain_probability_pct",
    "weather_source",
    "grid_position",
}
# Outcome-only columns: by construction the live path never has these
# (the race hasn't happened from its point of view) — not a skew finding.
OUTCOME_ONLY = {"finishing_position", "won", "points", "status", "dnf", "missing_qualifying_data"}

COMPARE_COLUMNS = [c for c in FEATURE_COLUMNS if c not in OUTCOME_ONLY] + ["grid_position", "weather_source", "weather_rain_probability_pct"]


async def skew_check_round(season: int, round_no: int, bundle: dict | None = None) -> dict:
    """Compare the live-reconstructed feature rows against the offline dataset for one round.

    The two driver rosters aren't guaranteed to match: the live path's
    roster comes from qualifying, the offline dataset's from race
    results, and a late substitute driver can start a race without
    having qualified (rare, but real — round 14 in this dataset has two).
    Only the driver_id intersection is compared column-by-column and for
    predicted probability; drivers present on only one side are reported
    separately rather than silently dropped or crashed on.
    """
    bundle = bundle if bundle is not None else load_model_bundle()

    live_df = await build_live_feature_rows(season, round_no)
    offline_df = pd.read_parquet(MODEL_DATASET_PATH)
    offline_df = offline_df[(offline_df["season"] == season) & (offline_df["round"] == round_no)]

    live_only = sorted(set(live_df["driver_id"]) - set(offline_df["driver_id"]))
    offline_only = sorted(set(offline_df["driver_id"]) - set(live_df["driver_id"]))
    common_drivers = sorted(set(live_df["driver_id"]) & set(offline_df["driver_id"]))

    live_indexed = live_df.set_index("driver_id").loc[common_drivers]
    offline_indexed = offline_df.set_index("driver_id").loc[common_drivers]

    column_diffs = []
    for col in sorted(set(COMPARE_COLUMNS) & set(live_indexed.columns) & set(offline_indexed.columns)):
        live_vals = live_indexed[col]
        offline_vals = offline_indexed[col]
        if pd.api.types.is_numeric_dtype(live_vals) and pd.api.types.is_numeric_dtype(offline_vals):
            differs = ~((live_vals - offline_vals).abs() < 1e-6) & ~(live_vals.isna() & offline_vals.isna())
        else:
            differs = live_vals.astype(str) != offline_vals.astype(str)
        n_diff = int(differs.sum())
        if n_diff:
            column_diffs.append(
                {
                    "column": col,
                    "rows_differing": n_diff,
                    "of_rows": len(common_drivers),
                    "expected": col in EXPECTED_TO_DIFFER,
                    "example_live": live_vals[differs].iloc[0],
                    "example_offline": offline_vals[differs].iloc[0],
                }
            )

    live_ranked = rank_predictions(bundle, live_df).set_index("driver_id")["win_probability"]
    offline_ranked = rank_predictions(bundle, offline_df).set_index("driver_id")["win_probability"]
    proba_diff = (live_ranked.loc[common_drivers] - offline_ranked.loc[common_drivers]).abs()

    return {
        "season": season,
        "round": round_no,
        "race_name": offline_df["race_name"].iloc[0] if len(offline_df) else live_df["race_name"].iloc[0],
        "n_drivers_common": len(common_drivers),
        "live_only_drivers": live_only,
        "offline_only_drivers": offline_only,
        "column_diffs": column_diffs,
        "max_probability_diff": float(proba_diff.max()),
        "mean_probability_diff": float(proba_diff.mean()),
        "live_top_pick": live_ranked.idxmax(),
        "offline_top_pick": offline_ranked.idxmax(),
        "top_pick_agrees": live_ranked.idxmax() == offline_ranked.idxmax(),
    }


def print_report(result: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"{result['season']} round {result['round']}: {result['race_name']} ({result['n_drivers_common']} drivers compared)")
    print("=" * 60)
    if result["live_only_drivers"] or result["offline_only_drivers"]:
        print(
            f"  Roster mismatch — live-only (qualified, not in offline?): {result['live_only_drivers']}; "
            f"offline-only (raced without qualifying, e.g. a late substitute): {result['offline_only_drivers']}"
        )
    if not result["column_diffs"]:
        print("  No column differences at all.")
    for diff in result["column_diffs"]:
        tag = "expected" if diff["expected"] else "UNEXPECTED"
        print(
            f"  [{tag:<10}] {diff['column']:<32} {diff['rows_differing']}/{diff['of_rows']} rows differ "
            f"(e.g. live={diff['example_live']!r} offline={diff['example_offline']!r})"
        )
    print(f"  Win-probability diff: max={result['max_probability_diff']:.4f} mean={result['mean_probability_diff']:.4f}")
    print(f"  Top pick — live: {result['live_top_pick']}  offline: {result['offline_top_pick']}  agree: {result['top_pick_agrees']}")


async def _main() -> None:
    bundle = load_model_bundle()
    rounds_to_check = [(2026, 14), (2026, 15), (2026, 16)]
    try:
        for season, round_no in rounds_to_check:
            result = await skew_check_round(season, round_no, bundle)
            print_report(result)

            unexpected = [d for d in result["column_diffs"] if not d["expected"]]
            if unexpected:
                print(f"  WARNING: unexpected column diffs found: {[d['column'] for d in unexpected]}")
    finally:
        await close_client()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()

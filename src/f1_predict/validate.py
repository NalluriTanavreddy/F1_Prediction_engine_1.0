"""Validation pass on the LightGBM result, run before building a prediction interface.

Report-only — does not retrain or overwrite models/model.pkl. Checks:

1. Overlap between LightGBM's top-1 hits and pole-sitter wins, plus a
   listing of the misses (model pick vs actual winner vs pole sitter).
2. Two trivial baselines next to LightGBM: logistic regression on grid
   position alone, and a non-probabilistic "winner is in the top 3 on the
   grid" rule.
3. Per-race probability normalization — see evaluate.py's
   normalize_per_race and log_loss_raw_unnormalized; not repeated here.
4. A walk-forward diagnostic: for each 2026 race, train on everything
   chronologically before it (2022-2025 + earlier 2026 rounds) and
   predict only that race, using LightGBM's already-tuned hyperparameters
   (no re-tuning per fold — this is a sanity check on the fixed-split
   result, not a new tuning run).
6. Where grid_position ranks in feature importance.

(Item 5 — dropping weather_rain_probability_pct — is a real change to
train.py's feature list, not a report, so it isn't in this file.)
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from f1_predict.evaluate import normalize_per_race, pole_sitter_win_rate, top_n_accuracy
from f1_predict.train import CATEGORICAL_FEATURES, FEATURE_COLUMNS, load_split, prepare_X_y, scale_pos_weight

MODELS_DIR = Path("models")


def load_lightgbm_bundle() -> dict:
    with open(MODELS_DIR / "lightgbm_model.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 1. Overlap between model hits and pole wins; list the misses
# ---------------------------------------------------------------------------


def overlap_and_misses(test_df: pd.DataFrame, proba_col: str = "proba") -> tuple[int, int, list[dict]]:
    hits_that_are_pole_wins = 0
    total_hits = 0
    misses = []

    for (season, round_no), race in test_df.groupby(["season", "round"]):
        actual_winner = race.loc[race["won"] == 1].iloc[0]
        model_pick = race.loc[race[proba_col].idxmax()]
        pole_sitter = race.loc[race["quali_position"] == 1].iloc[0] if (race["quali_position"] == 1).any() else None

        is_hit = model_pick["driver_id"] == actual_winner["driver_id"]
        if is_hit:
            total_hits += 1
            if pole_sitter is not None and pole_sitter["driver_id"] == actual_winner["driver_id"]:
                hits_that_are_pole_wins += 1
        else:
            misses.append(
                {
                    "season": int(season),
                    "round": int(round_no),
                    "race_name": race["race_name"].iloc[0],
                    "model_picked": model_pick["driver_code"],
                    "actual_winner": actual_winner["driver_code"],
                    "pole_sitter": pole_sitter["driver_code"] if pole_sitter is not None else "none",
                }
            )

    return hits_that_are_pole_wins, total_hits, misses


# ---------------------------------------------------------------------------
# 2. Simple baselines: logistic regression on grid position; grid-top-3 rule
# ---------------------------------------------------------------------------


def grid_logistic_regression_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_rows = train_df.dropna(subset=["grid_position"])
    test_rows = test_df.dropna(subset=["grid_position"]).copy()

    X_train = train_rows[["grid_position"]].astype("float64")
    y_train = train_rows["won"].astype(int)
    X_test = test_rows[["grid_position"]].astype("float64")

    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)
    test_rows["proba"] = model.predict_proba(X_test)[:, 1]
    test_rows["proba_norm"] = normalize_per_race(test_rows, "proba")

    return {
        "top1_accuracy": top_n_accuracy(test_rows, "proba", 1),
        "top3_accuracy": top_n_accuracy(test_rows, "proba", 3),
        "log_loss": log_loss(test_rows["won"], test_rows["proba_norm"]),
    }


def grid_top3_rule_baseline(test_df: pd.DataFrame) -> float:
    """Non-probabilistic rule: predict the winner is one of the top 3 grid slots.

    Only a top-3 number is meaningful for a deterministic rule like this —
    there's no single "top-1 pick" or probability to score with log loss.
    (Grid P1's win rate as a top-1 baseline is already the pole baseline
    in evaluate.py.)
    """
    hits = 0
    races = 0
    for _, race in test_df.groupby(["season", "round"]):
        races += 1
        winner = race.loc[race["won"] == 1].iloc[0]
        if pd.notna(winner["grid_position"]) and winner["grid_position"] <= 3:
            hits += 1
    return hits / races


# ---------------------------------------------------------------------------
# 4. Walk-forward diagnostic (report-only; does not touch models/model.pkl)
# ---------------------------------------------------------------------------


def walk_forward_diagnostic(full_df: pd.DataFrame, lgbm_params: dict) -> pd.DataFrame:
    """For each 2026 round, train on everything strictly before it and predict it.

    Uses LightGBM's already-tuned hyperparameters as-is (no re-tuning per
    fold — that would be 16 full tuning runs for a sanity check). The
    first fold's training set is exactly the normal 2022-2025 training
    set; each subsequent fold adds the previous folds' 2026 rounds too.
    """
    test_rounds = sorted(full_df.loc[full_df["season"] == 2026, "round"].unique())
    predictions = []

    for round_no in test_rounds:
        train_fold = full_df[(full_df["season"] < 2026) | ((full_df["season"] == 2026) & (full_df["round"] < round_no))]
        predict_fold = full_df[(full_df["season"] == 2026) & (full_df["round"] == round_no)]

        X_train, y_train = prepare_X_y(train_fold)
        X_pred, _ = prepare_X_y(predict_fold)

        model = lgb.LGBMClassifier(
            objective="binary",
            scale_pos_weight=scale_pos_weight(y_train),
            random_state=42,
            verbosity=-1,
            **lgbm_params,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

        fold_result = predict_fold.copy()
        fold_result["proba"] = model.predict_proba(X_pred)[:, 1]
        predictions.append(fold_result)

    return pd.concat(predictions, ignore_index=True)


# ---------------------------------------------------------------------------
# 6. Grid position's importance rank
# ---------------------------------------------------------------------------


def grid_position_rank(bundle: dict) -> str:
    importances = bundle["model"].feature_importances_
    ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda t: t[1], reverse=True)
    rank = next(i for i, (f, _) in enumerate(ranked, start=1) if f == "grid_position")
    value = next(v for f, v in ranked if f == "grid_position")
    return f"grid_position ranks {rank} of {len(ranked)} by importance (value={value:.1f})"


def main() -> None:
    train_df, test_df = load_split()
    bundle = load_lightgbm_bundle()

    test_scored = test_df.copy()
    X_test, _ = prepare_X_y(test_scored)
    test_scored["proba"] = bundle["model"].predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("1. OVERLAP: LightGBM top-1 hits vs pole-sitter wins")
    print("=" * 60)
    pole_hits, total_hits, misses = overlap_and_misses(test_scored)
    print(f"  {total_hits} LightGBM top-1 hits total")
    print(f"  {pole_hits} of those {total_hits} hits are races the pole-sitter also won")
    print(f"  {total_hits - pole_hits} hits where the model correctly picked a non-pole winner")
    print(f"\n  The {len(misses)} misses (model pick / actual winner / pole sitter):")
    for m in misses:
        print(f"    {m['season']} R{m['round']} {m['race_name']:<28} model={m['model_picked']:<4} actual={m['actual_winner']:<4} pole={m['pole_sitter']}")

    print(f"\n{'=' * 60}")
    print("2. SIMPLE BASELINES vs LightGBM (same 16 test races)")
    print("=" * 60)
    logreg = grid_logistic_regression_baseline(train_df, test_df)
    grid_top3 = grid_top3_rule_baseline(test_df)
    lgbm_top1 = top_n_accuracy(test_scored, "proba", 1)
    lgbm_top3 = top_n_accuracy(test_scored, "proba", 3)
    test_scored["proba_norm"] = normalize_per_race(test_scored, "proba")
    lgbm_logloss = log_loss(test_scored["won"], test_scored["proba_norm"])
    print(f"  {'model':<28} {'top-1':>8} {'top-3':>8} {'log loss':>10}")
    print(f"  {'LightGBM':<28} {lgbm_top1:>8.1%} {lgbm_top3:>8.1%} {lgbm_logloss:>10.4f}")
    print(f"  {'logistic regression (grid)':<28} {logreg['top1_accuracy']:>8.1%} {logreg['top3_accuracy']:>8.1%} {logreg['log_loss']:>10.4f}")
    print(f"  {'grid top-3 rule':<28} {'n/a':>8} {grid_top3:>8.1%} {'n/a':>10}")

    print(f"\n{'=' * 60}")
    print("3. PER-RACE NORMALIZATION")
    print("=" * 60)
    print("  Confirmed NOT normalized in the original evaluate.py run — model")
    print("  predict_proba output has no reason to sum to 1 within a race, since")
    print("  the model is trained as an independent per-row binary classifier.")
    print("  Fixed: evaluate.py now normalizes per race before computing the")
    print("  log loss compared against the pole baseline (which IS already")
    print("  built to sum to 1 per race by construction) and reports the raw,")
    print("  unnormalized log loss alongside it for transparency.")
    sums = test_scored.groupby(["season", "round"])["proba"].sum()
    print(f"  Raw per-race probability sums before normalizing: min={sums.min():.2f} mean={sums.mean():.2f} max={sums.max():.2f}")

    print(f"\n{'=' * 60}")
    print("4. WALK-FORWARD DIAGNOSTIC (report-only, no model/file changes)")
    print("=" * 60)
    wf = walk_forward_diagnostic(pd.concat([train_df, test_df], ignore_index=True), bundle["params"])
    wf["proba_norm"] = normalize_per_race(wf, "proba")
    wf_top1 = top_n_accuracy(wf, "proba", 1)
    wf_top3 = top_n_accuracy(wf, "proba", 3)
    wf_logloss = log_loss(wf["won"], wf["proba_norm"])
    print(f"  {'':<28} {'top-1':>8} {'top-3':>8} {'log loss':>10}")
    print(f"  {'Fixed split (evaluate.py)':<28} {lgbm_top1:>8.1%} {lgbm_top3:>8.1%} {lgbm_logloss:>10.4f}")
    print(f"  {'Walk-forward (this check)':<28} {wf_top1:>8.1%} {wf_top3:>8.1%} {wf_logloss:>10.4f}")

    print(f"\n{'=' * 60}")
    print("6. GRID POSITION FEATURE IMPORTANCE")
    print("=" * 60)
    print(f"  {grid_position_rank(bundle)}")


if __name__ == "__main__":
    main()

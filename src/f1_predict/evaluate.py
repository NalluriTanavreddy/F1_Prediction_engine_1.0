"""Evaluate the trained LightGBM/XGBoost models on the 2026 holdout.

Loads models/{lightgbm,xgboost}_model.pkl, scores the 2026 test set,
computes top-1/top-3 accuracy, log loss, and the pole-position baseline,
reports feature importance (flagging where racecraft tendency ranks),
picks the model with the better test log loss as "production" (saved as
models/model.pkl), and writes results/metrics.json.

Top-1/top-3 accuracy are computed per RACE (argmax / top-3 of predicted
win probability among that race's drivers), not per row — the models are
trained as per-row binary classifiers (as asked), but "did we pick the
winner" is inherently a per-race question.

Log loss is computed on probabilities normalized to sum to 1 within each
race (see normalize_per_race) — the raw per-row predict_proba output
isn't constrained to do that, since the model has no notion of "race" at
fit time, while the pole baseline below *is* built to sum to 1 per race.
Comparing the model's raw, unnormalized log loss against that baseline
would be comparing two different things; log_loss_raw_unnormalized is
kept in the output for transparency but log_loss (normalized) is the
number to trust against the pole baseline.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss

from f1_predict.train import FEATURE_COLUMNS, load_split, prepare_X_y

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def load_model(name: str) -> dict:
    with open(MODELS_DIR / f"{name}_model.pkl", "rb") as f:
        return pickle.load(f)


def predict_proba(bundle: dict, df: pd.DataFrame) -> pd.Series:
    X, _ = prepare_X_y(df)
    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index)


def normalize_per_race(df: pd.DataFrame, proba_col: str, season_round_cols: tuple[str, str] = ("season", "round")) -> pd.Series:
    """Rescale predicted probabilities so each race's field sums to 1.

    The model is trained as an independent per-row binary classifier (one
    win/not-win probability per driver), so nothing forces a race's raw
    probabilities to sum to 1 the way a proper per-race probability
    distribution would. The pole baseline *is* constructed to sum to 1
    (pole gets the pole rate, the rest split the remainder), so comparing
    the model's raw log loss against it isn't quite apples-to-apples.
    Normalizing per race fixes that for log loss; it doesn't change
    top-1/top-3 since those only depend on the within-race ranking, which
    a per-race rescaling preserves.
    """
    season_col, round_col = season_round_cols
    sums = df.groupby([season_col, round_col])[proba_col].transform("sum")
    return df[proba_col] / sums


def top_n_accuracy(df: pd.DataFrame, proba_col: str, n: int) -> float:
    """Fraction of races where the actual winner is among the top-n predicted probabilities."""
    hits = 0
    races = 0
    for _, race in df.groupby(["season", "round"]):
        races += 1
        top_n_drivers = race.nlargest(n, proba_col)
        if race.loc[race["won"] == 1, proba_col].index.isin(top_n_drivers.index).any():
            hits += 1
    return hits / races


def pole_sitter_win_rate(df: pd.DataFrame) -> float:
    pole_rows = df[df["quali_position"].notna() & (df["quali_position"] == 1)]
    return float(pole_rows["won"].mean())


def pole_baseline_logloss(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    """A proper probabilistic baseline for log-loss comparison, not just a win-rate.

    Assigns P(win) = (training pole win rate) to the pole sitter in each
    test race, and splits the remaining probability uniformly across the
    rest of that race's field. Not explicitly requested, but cheap to add
    and a fairer log-loss comparison than reporting accuracy alone.
    """
    pole_rate = pole_sitter_win_rate(train_df)
    proba = []
    y_true = []
    for _, race in test_df.groupby(["season", "round"]):
        field_size = len(race)
        other_prob = (1 - pole_rate) / max(field_size - 1, 1)
        for _, row in race.iterrows():
            is_pole = pd.notna(row["quali_position"]) and row["quali_position"] == 1
            proba.append(pole_rate if is_pole else other_prob)
            y_true.append(row["won"])
    return log_loss(y_true, proba)


def feature_importance(bundle: dict, model_name: str) -> list[dict]:
    model = bundle["model"]
    importances = model.feature_importances_
    ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda t: t[1], reverse=True)
    return [{"feature": f, "importance": float(v)} for f, v in ranked]


def racecraft_rank_note(importance_list: list[dict]) -> str:
    rank = next(i for i, item in enumerate(importance_list, start=1) if item["feature"] == "driver_racecraft_avg_delta")
    total = len(importance_list)
    return f"driver_racecraft_avg_delta ranks {rank} of {total} by importance"


def evaluate_model(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    bundle = load_model(name)
    test_df = test_df.copy()
    test_df["proba"] = predict_proba(bundle, test_df)
    test_df["proba_norm"] = normalize_per_race(test_df, "proba")

    top1 = top_n_accuracy(test_df, "proba", 1)
    top3 = top_n_accuracy(test_df, "proba", 3)
    raw_loss = log_loss(test_df["won"], test_df["proba"])
    loss = log_loss(test_df["won"], test_df["proba_norm"])
    importance = feature_importance(bundle, name)

    return {
        "model": name,
        "params": bundle["params"],
        "top1_accuracy": top1,
        "top3_accuracy": top3,
        "log_loss": loss,
        "log_loss_raw_unnormalized": raw_loss,
        "feature_importance_top10": importance[:10],
        "racecraft_tendency_rank": racecraft_rank_note(importance),
    }


def main() -> None:
    train_df, test_df = load_split()
    n_test_races = test_df.drop_duplicates(subset=["season", "round"]).shape[0]

    pole_rate_train = pole_sitter_win_rate(train_df)
    pole_rate_test = pole_sitter_win_rate(test_df)
    pole_logloss_test = pole_baseline_logloss(train_df, test_df)

    print("=" * 60)
    print("POLE-POSITION BASELINE")
    print("=" * 60)
    print(f"  Pole-sitter win rate, training data (2022-2025):        {pole_rate_train:.1%}")
    print(f"  Pole-sitter win rate, test data (2026, {n_test_races} races):      {pole_rate_test:.1%}")
    print(f"  Pole baseline log loss on test set:                     {pole_logloss_test:.4f}")
    print(f"  ({n_test_races} test races is small — top-1/top-3 will be noisy; log loss and")
    print("   this baseline comparison are the more robust numbers to trust.)")

    results = {}
    for name in ("lightgbm", "xgboost"):
        print(f"\n{'=' * 60}")
        print(f"{name.upper()}")
        print("=" * 60)
        result = evaluate_model(name, train_df, test_df)
        results[name] = result
        top1_hits = int(round(result["top1_accuracy"] * n_test_races))
        top3_hits = int(round(result["top3_accuracy"] * n_test_races))
        print(f"  Top-1 accuracy: {result['top1_accuracy']:.1%}  ({top1_hits}/{n_test_races} races)")
        print(f"  Top-3 accuracy: {result['top3_accuracy']:.1%}  ({top3_hits}/{n_test_races} races)")
        print(f"  Log loss (per-race normalized): {result['log_loss']:.4f}  (pole baseline: {pole_logloss_test:.4f})")
        print(f"  Log loss (raw, unnormalized):   {result['log_loss_raw_unnormalized']:.4f}  (not comparable to the baseline above)")
        print(f"  {result['racecraft_tendency_rank']}")
        print("  Top 5 features:")
        for item in result["feature_importance_top10"][:5]:
            print(f"    {item['feature']:<36} {item['importance']:.1f}")

    winner = min(results, key=lambda n: results[n]["log_loss"])
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION: keep {winner} (lower test log loss)")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / f"{winner}_model.pkl", "rb") as f:
        winning_bundle = pickle.load(f)
    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(winning_bundle, f)
    print(f"Saved {winner}'s model as models/model.pkl (the canonical model for inference)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "pole_baseline": {
            "win_rate_train": pole_rate_train,
            "win_rate_test": pole_rate_test,
            "log_loss_test": pole_logloss_test,
        },
        "models": results,
        "recommendation": {
            "production_model": winner,
            "reason": "lower log loss on the 2026 test set — the more robust metric given only 16 test races",
        },
        "weather_eval_caveat": (
            "Test-set weather uses actual historical observations (races already "
            "happened), not a pre-race forecast. These metrics measure accuracy "
            "if weather were known perfectly, not true live-deployment accuracy."
        ),
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()

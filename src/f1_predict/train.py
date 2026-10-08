"""Train LightGBM and XGBoost win-probability classifiers.

Reads model_dataset.parquet, tunes each model with a modest Optuna budget
(validating on 2025 — the most recent pre-2026 season — so the 2026 test
set is never touched during tuning), retrains on the full 2022-2025
training set with the tuned params, and saves both models under models/.

Target imbalance: ~1 winner per ~20-car field, so scale_pos_weight
(XGBoost) / the equivalent LightGBM param corrects for it on both models,
computed from the training set's actual win rate.

2026-era sample weighting: NOT applied. All 352 2026 rows are in the test
set and none are in train (see README) — there is no 2026-era training
row to upweight. Confirmed with the user rather than moving 2026 races
into a walk-forward split, which would have shrunk the already-small
16-race test set. Revisit once a real chunk of 2026 becomes historical.

weather_rain_probability_pct is deliberately excluded from the feature
list: it's forecast-only (see clients/open_meteo.py), so every row in the
current, all-historical dataset has it null. A column that's 100% null in
training carries zero signal and the model can't learn anything about its
future, populated behavior from training data that never has it populated.
weather_precip_mm already exists in both the historical-actual and
forecast paths and captures the same "how much rain" signal, so dropping
rain_probability_pct rather than inventing a historical substitute for it
loses nothing.
"""

import pickle
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATASET_PATH = Path("data/processed/model_dataset.parquet")
MODELS_DIR = Path("models")
N_TRIALS = 25
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "season",
    "round",
    "quali_position",
    "grid_position",
    "q1_time_s",
    "q2_time_s",
    "q3_time_s",
    "best_quali_time_s",
    "gap_to_pole_s",
    "sprint_position",
    "sprint_points",
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
    "weather_temp_max_c",
    "weather_precip_mm",
    "weather_wind_speed_max_kph",
]
CATEGORICAL_FEATURES = ["driver_id", "constructor_id", "circuit_id", "era"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "won"


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATASET_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return train_df, test_df


def prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLUMNS].copy()
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df[TARGET].astype(int)
    return X, y


def tune_validation_split(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set out of TRAINING data only, for tuning.

    2025 (the most recent pre-2026 season) is validation; 2022-2024 is the
    tuning-train set. The 2026 test set is never touched during tuning.
    """
    tune_train = train_df[train_df["season"] < 2025].reset_index(drop=True)
    val = train_df[train_df["season"] == 2025].reset_index(drop=True)
    return tune_train, val


def scale_pos_weight(y: pd.Series) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    return neg / pos


def tune_lightgbm(tune_train: pd.DataFrame, val: pd.DataFrame) -> dict:
    X_tune, y_tune = prepare_X_y(tune_train)
    X_val, y_val = prepare_X_y(val)
    spw = scale_pos_weight(y_tune)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "scale_pos_weight": spw,
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tune, y_tune, categorical_feature=CATEGORICAL_FEATURES)
        proba = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, proba)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def tune_xgboost(tune_train: pd.DataFrame, val: pd.DataFrame) -> dict:
    X_tune, y_tune = prepare_X_y(tune_train)
    X_val, y_val = prepare_X_y(val)
    spw = scale_pos_weight(y_tune)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "scale_pos_weight": spw,
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "enable_categorical": True,
            "tree_method": "hist",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tune, y_tune)
        proba = model.predict_proba(X_val)[:, 1]
        return log_loss(y_val, proba)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def train_final_lightgbm(train_df: pd.DataFrame, best_params: dict) -> lgb.LGBMClassifier:
    X, y = prepare_X_y(train_df)
    model = lgb.LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight(y),
        random_state=RANDOM_STATE,
        verbosity=-1,
        **best_params,
    )
    model.fit(X, y, categorical_feature=CATEGORICAL_FEATURES)
    return model


def train_final_xgboost(train_df: pd.DataFrame, best_params: dict) -> xgb.XGBClassifier:
    X, y = prepare_X_y(train_df)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight(y),
        random_state=RANDOM_STATE,
        enable_categorical=True,
        tree_method="hist",
        **best_params,
    )
    model.fit(X, y)
    return model


def main() -> None:
    train_df, test_df = load_split()
    tune_train, val = tune_validation_split(train_df)
    print(f"Tuning-train: {len(tune_train)} rows (2022-2024)  Validation: {len(val)} rows (2025)")
    print(f"Final train: {len(train_df)} rows (2022-2025)  Test: {len(test_df)} rows (2026, untouched until evaluate.py)")

    print(f"\nTuning LightGBM ({N_TRIALS} trials)...")
    lgb_params = tune_lightgbm(tune_train, val)
    print(f"  best params: {lgb_params}")

    print(f"\nTuning XGBoost ({N_TRIALS} trials)...")
    xgb_params = tune_xgboost(tune_train, val)
    print(f"  best params: {xgb_params}")

    print("\nTraining final models on full 2022-2025 training set...")
    lgb_model = train_final_lightgbm(train_df, lgb_params)
    xgb_model = train_final_xgboost(train_df, xgb_params)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_DIR / "lightgbm_model.pkl", "wb") as f:
        pickle.dump({"model": lgb_model, "feature_columns": FEATURE_COLUMNS, "params": lgb_params}, f)
    with open(MODELS_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump({"model": xgb_model, "feature_columns": FEATURE_COLUMNS, "params": xgb_params}, f)

    print(f"\nSaved models/lightgbm_model.pkl and models/xgboost_model.pkl")
    print("Run `python -m f1_predict.evaluate` to compare them on the 2026 test set.")


if __name__ == "__main__":
    main()

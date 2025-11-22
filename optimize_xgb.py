# optimize_xgb.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src import config as cf

SPLITS = Path(cf.SPLITS_DIR)
MODELS = Path(cf.MODELS_DIR)
MAX_GAP = cf.XGB_MAX_GAP_SECONDS


def reorder_memberships(U):
    """Convert FCM output [clean, high, medium] → [clean, medium, high]."""
    return np.column_stack([U[:, 2], U[:, 0], U[:, 1]])


def load_continuous_pairs(Z, U, csv_path):
    """Return (X, Y) for continuous t→t+1 pairs only."""
    df = pd.read_csv(csv_path)
    start = pd.to_datetime(df["start_time"])
    end   = pd.to_datetime(df["end_time"])
    loc   = df["location"].values

    assert len(df) == len(Z) == len(U), "Metadata and arrays must match."

    X, Y = [], []
    for i in range(len(df) - 1):
        if loc[i] != loc[i + 1]:
            continue
        gap_sec = (start[i + 1] - end[i]).total_seconds()
        if abs(gap_sec) <= MAX_GAP:
            X.append(Z[i])      # embedding at t
            Y.append(U[i + 1])  # membership at t+1

    return np.array(X), np.array(Y)


def load_data():
    # embeddings
    Z_tr = np.load(SPLITS / "train_embeddings_pca.npy")
    Z_va = np.load(SPLITS / "val_embeddings_pca.npy")

    # memberships
    U_tr = reorder_memberships(np.load(SPLITS / "train_memberships.npy"))
    U_va = reorder_memberships(np.load(SPLITS / "val_memberships.npy"))

    # continuous transitions
    X_tr, Y_tr = load_continuous_pairs(Z_tr, U_tr, SPLITS / "train_metadata.csv")
    X_va, Y_va = load_continuous_pairs(Z_va, U_va, SPLITS / "val_metadata.csv")

    print(f"Train continuous pairs (opt): {len(X_tr)}")
    print(f"Val   continuous pairs (opt): {len(X_va)}")

    return X_tr, Y_tr, X_va, Y_va


X_TR, Y_TR, X_VA, Y_VA = load_data()


def make_objective(y_tr, y_va):
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="reg:squarederror",
            tree_method="hist",
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 20.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            n_jobs=-1,
            random_state=42,
        )

        model = XGBRegressor(**params)
        model.fit(X_TR, y_tr, eval_set=[(X_VA, y_va)], verbose=False)

        pred = model.predict(X_VA)
        r2 = r2_score(y_va, pred)
        return 1.0 - r2  # minimize 1 - R²
    return objective


def tune_one(name: str, col_idx: int, n_trials: int = 50):
    y_tr = Y_TR[:, col_idx]
    y_va = Y_VA[:, col_idx]

    print(f"\n=== Optimizing {name.upper()} (col {col_idx}) ===")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(y_tr, y_va), n_trials=n_trials)

    best_loss = study.best_value
    best_r2 = 1.0 - best_loss
    print(f"Best R² ({name}): {best_r2:.4f}")
    print("Best params:", study.best_params)

    best_params = study.best_params
    best_params.update(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    MODELS.mkdir(exist_ok=True)
    out_path = MODELS / f"xgb_{name}_params.json"
    out_path.write_text(json.dumps(best_params, indent=2))
    print(f"Saved best params to {out_path}")


def main():
    # columns in Y: 0=clean, 1=medium, 2=high
    tune_one("clean",  0)
    tune_one("medium", 1)
    tune_one("high",   2)


if __name__ == "__main__":
    main()

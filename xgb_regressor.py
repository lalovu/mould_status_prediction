# xgb_regressor.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import spearmanr

from src import config as cf

SPLITS = Path(cf.SPLITS_DIR)
MODELS = Path(cf.MODELS_DIR)
MAX_GAP = cf.XGB_MAX_GAP_SECONDS


def reorder_memberships(U):
    """Convert FCM output [clean, high, medium] → [clean, medium, high]."""
    return np.column_stack([U[:, 2], U[:, 0], U[:, 1]])


def load_continuous_pairs(Z, U, csv_path):
    """Return (X, Y) pairs only where next window is truly continuous."""
    df = pd.read_csv(csv_path)
    start = pd.to_datetime(df["start_time"])
    end   = pd.to_datetime(df["end_time"])
    loc   = df["location"].values

    X, Y = [], []
    for i in range(len(df) - 1):
        if loc[i] != loc[i + 1]:
            continue

        gap_sec = (start[i + 1] - end[i]).total_seconds()
        if abs(gap_sec) <= MAX_GAP:
            X.append(Z[i])
            Y.append(U[i + 1])

    return np.array(X), np.array(Y)


def load_params(name):
    path = MODELS / f"xgb_{name}_params.json"
    if path.exists():
        return json.loads(path.read_text())

    # simple fallback defaults
    return dict(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1.0,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=42,
    )


def evaluate(name, model, X, y):
    p = model.predict(X)
    mae = mean_absolute_error(y, p)
    rmse = root_mean_squared_error(y, p)
    r2 = r2_score(y, p)
    rho, _ = spearmanr(y.ravel(), p.ravel(), nan_policy="omit")

    print(f"\n{name}")
    print(f"  MAE:      {mae:.4f}")
    print(f"  RMSE:     {rmse:.4f}")
    print(f"  R²:       {r2:.4f}")
    print(f"  Spearman: {rho:.4f}")


def main():
    # embeddings
    Z_tr = np.load(SPLITS / "train_embeddings_pca.npy")
    Z_va = np.load(SPLITS / "val_embeddings_pca.npy")
    Z_te = np.load(SPLITS / "test_embeddings_pca.npy")

    # memberships
    U_tr = reorder_memberships(np.load(SPLITS / "train_memberships.npy"))
    U_va = reorder_memberships(np.load(SPLITS / "val_memberships.npy"))
    U_te = reorder_memberships(np.load(SPLITS / "test_memberships.npy"))

    # continuous transitions only
    X_tr, Y_tr = load_continuous_pairs(Z_tr, U_tr, SPLITS / "train_metadata.csv")
    X_va, Y_va = load_continuous_pairs(Z_va, U_va, SPLITS / "val_metadata.csv")
    X_te, Y_te = load_continuous_pairs(Z_te, U_te, SPLITS / "test_metadata.csv")

    print(f"Train continuous pairs: {len(X_tr)}")
    print(f"Val   continuous pairs: {len(X_va)}")
    print(f"Test  continuous pairs: {len(X_te)}")

    # column targets
    yc_tr, ym_tr, yh_tr = Y_tr.T
    yc_va, ym_va, yh_va = Y_va.T
    yc_te, ym_te, yh_te = Y_te.T

    # params
    p_clean  = load_params("clean")
    p_medium = load_params("medium")
    p_high   = load_params("high")

    # train models
    m_clean  = XGBRegressor(**p_clean).fit(X_tr, yc_tr, eval_set=[(X_va, yc_va)], verbose=False)
    m_medium = XGBRegressor(**p_medium).fit(X_tr, ym_tr, eval_set=[(X_va, ym_va)], verbose=False)
    m_high   = XGBRegressor(**p_high).fit(X_tr, yh_tr, eval_set=[(X_va, yh_va)], verbose=False)

    # evaluate
    print("\n=== TEST PERFORMANCE (continuous transitions) ===")
    evaluate("CLEAN ",  m_clean,  X_te, yc_te)
    evaluate("MEDIUM",  m_medium, X_te, ym_te)
    evaluate("HIGH  ",  m_high,   X_te, yh_te)

    # save
    MODELS.mkdir(exist_ok=True)
    m_clean.save_model(MODELS / "xgb_clean.json")
    m_medium.save_model(MODELS / "xgb_medium.json")
    m_high.save_model(MODELS / "xgb_high.json")


if __name__ == "__main__":
    main()

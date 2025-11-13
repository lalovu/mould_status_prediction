import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

ORDER = [2, 0, 1]  # original FCM cols -> [Clean, Medium, High]

def reorder_memberships(U, order=ORDER):
    # U: (N, 3) in original FCM order
    return U[:, order]

E = np.load("splits/train_embeddings.npy")
U = np.load("splits/train_memberships.npy")
E_val = np.load("splits/val_embeddings.npy")
U_val = np.load("splits/val_memberships.npy")

U = reorder_memberships(U)
U_val = reorder_memberships(U_val)

X_train, y_train = E[:-1], U[1:]
X_val, y_val = E_val[:-1], U_val[1:]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "objective": "reg:squarederror",
        "random_state": 0,
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("\nBest parameters:")
print(study.best_params)
print(f"Best validation MSE: {study.best_value:.5f}")

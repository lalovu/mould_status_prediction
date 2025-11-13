from xgboost import XGBRegressor
import numpy as np

E_tr = np.load("splits/train_embeddings.npy")
U_tr = np.load("splits/train_memberships.npy")
E_va = np.load("splits/val_embeddings.npy")
U_va = np.load("splits/val_memberships.npy")

X_train = np.vstack([E_tr[:-1], E_va[:-1]])
y_train = np.vstack([U_tr[1:],  U_va[1:]])

best_params = {
    'n_estimators': 227,
    'max_depth': 8,
    'learning_rate': 0.0100944533714591,
    'subsample': 0.9808908845341231,
    'colsample_bytree': 0.9713185992354655,
    'min_child_weight': 6.894728144326067,
    'reg_lambda': 0.0012056156390197951,
    'reg_alpha': 0.06809726552705449,
    'objective': 'reg:squarederror',
    'random_state': 0,
    'n_jobs': -1
}

model = XGBRegressor(**best_params)
model.fit(X_train, y_train)
model.save_model("splits/xgb_forecaster_best.json")
print("✅ Final XGBoost forecaster saved.")

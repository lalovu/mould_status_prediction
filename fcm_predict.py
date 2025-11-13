# fcm_and_predict.py
import numpy as np
from skfuzzy.cluster import cmeans, cmeans_predict

SPLITS = "splits"
m = 2.0          # fuzziness (standard)
C = 3            # clean / medium / high

# 1) load
tr = np.load(f"{SPLITS}/train_embeddings.npy")
va = np.load(f"{SPLITS}/val_embeddings.npy")
te = np.load(f"{SPLITS}/test_embeddings.npy")

print(f"Train shape: {tr.shape}")
print(f"Val shape: {va.shape}")
print(f"Test shape: {te.shape}")

# 2) fit FCM on TRAIN
# skfuzzy expects (features, samples)
cntr, u_tr, u0, d, jm, p, fpc = cmeans(
    data=tr.T, c=C, m=m, error=1e-5, maxiter=200, init=None
)

print(f"FCM converged. FPC: {fpc:.4f}")

# 3) predict memberships for VAL/TEST with learned centers
u_va, u0_va, d_va, jm_va, p_va, fpc_va = cmeans_predict(
    va.T, cntr, m, error=1e-5, maxiter=200
)
u_te, u0_te, d_te, jm_te, p_te, fpc_te = cmeans_predict(
    te.T, cntr, m, error=1e-5, maxiter=200
)

# sanity: memberships sum to 1
assert np.allclose(u_tr.sum(axis=0), 1, atol=1e-5), "Train memberships don't sum to 1"
assert np.allclose(u_va.sum(axis=0), 1, atol=1e-5), "Val memberships don't sum to 1"
assert np.allclose(u_te.sum(axis=0), 1, atol=1e-5), "Test memberships don't sum to 1"

print("Membership assertions passed ✓")

# Save results
np.save(f"{SPLITS}/fcm_centers.npy", cntr)
np.save(f"{SPLITS}/train_memberships.npy", u_tr.T)  # (N_tr, 3)
np.save(f"{SPLITS}/val_memberships.npy",   u_va.T)  # (N_va, 3)
np.save(f"{SPLITS}/test_memberships.npy",  u_te.T)  # (N_te, 3)

print(f"\nFCM complete:")
print(f"  Train: {u_tr.T.shape}")
print(f"  Val:   {u_va.T.shape}")
print(f"  Test:  {u_te.T.shape}")
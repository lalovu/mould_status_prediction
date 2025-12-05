import numpy as np
from skfuzzy.cluster import cmeans, cmeans_predict

SPLITS = "splits"
N_CLUSTERS = 3
M = 2.0
ERROR = 1e-5
MAXITER = 150
SEED = 42


def load_embeddings(split: str) -> np.ndarray:
    arr = np.load(f"{SPLITS}/{split}_embeddings_pca_test.npy")
    if arr.ndim != 2:
        raise ValueError(f"{split} embeddings must be 2D, got {arr.shape}")
    return arr


def main():
    # load
    tr = load_embeddings("train")
    va = load_embeddings("val")
    te = load_embeddings("test")

    print("Embeddings:")
    print("  train:", tr.shape)
    print("  val  :", va.shape)
    print("  test :", te.shape)

    # fit FCM on train
    cntr, u_tr, _, _, _, _, fpc = cmeans(
        tr.T,
        c=N_CLUSTERS,
        m=M,
        error=ERROR,
        maxiter=MAXITER,
        init=None,
        seed=SEED,
    )
    print(f"FPC: {fpc:.4f}")

    # predict memberships for val/test
    def predict(x: np.ndarray) -> np.ndarray:
        u, _, _, _, _, _ = cmeans_predict(
            test_data=x.T,
            cntr_trained=cntr,
            m=M,
            error=ERROR,
            maxiter=MAXITER,
        )
        return u.T  # (n_samples, n_clusters)

    u_tr = u_tr.T
    u_va = predict(va)
    u_te = predict(te)

    for name, u in [("train", u_tr), ("val", u_va), ("test", u_te)]:
        sums = u.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-5):
            raise ValueError(f"{name} memberships rows do not sum to 1")

    np.save(f"{SPLITS}/fcm_centers_test.npy", cntr)
    np.save(f"{SPLITS}/train_memberships_test.npy", u_tr)
    np.save(f"{SPLITS}/val_memberships_test.npy", u_va)
    np.save(f"{SPLITS}/test_memberships_test.npy", u_te)

    print("Saved:")
    print("  fcm_centers.npy")
    print("  train_memberships.npy")
    print("  val_memberships.npy")
    print("  test_memberships.npy")


if __name__ == "__main__":
    main()

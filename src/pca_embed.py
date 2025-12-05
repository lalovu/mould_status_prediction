# pca.py
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.stats import chi2
from joblib import dump
from src.utils import plot_pca_variance



def load_embeddings(splits_path: Path):
    """Load LSTM embeddings for train/val/test."""
    train = np.load(splits_path / "train_embeddings_test.npy")
    val   = np.load(splits_path / "val_embeddings_test.npy")
    test  = np.load(splits_path / "test_embeddings_test.npy")
    return train, val, test


def corr_matrix(X):
    return np.corrcoef(X, rowvar=False)


def kmo(X):
    """Kaiser-Meyer-Olkin measure."""
    R = corr_matrix(X)
    invR = np.linalg.inv(R)

    A = -invR / np.sqrt(np.outer(np.diag(invR), np.diag(invR)))
    np.fill_diagonal(A, 0.0)

    R2 = R ** 2
    A2 = A ** 2
    np.fill_diagonal(R2, 0.0)

    sum_r = R2.sum()
    sum_a = A2.sum()
    return sum_r / (sum_r + sum_a)


def bartlett(X):
    """Bartlett's test of sphericity."""
    n, p = X.shape
    R = corr_matrix(X)
    detR = np.linalg.det(R)
    if detR <= 0:
        return np.nan, np.nan, np.nan

    chi_sq = -(n - 1 - (2 * p + 5) / 6) * np.log(detR)
    df = p * (p - 1) / 2
    p_val = 1 - chi2.cdf(chi_sq, df)
    return chi_sq, df, p_val


def run_pca_on_embeddings(train, val, test, n_components: int):
    """Fit PCA on train embeddings and transform all splits."""
    pca = PCA(n_components=n_components, random_state=0)
    train_p = pca.fit_transform(train)
    val_p   = pca.transform(val)
    test_p  = pca.transform(test)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} comps, explained={explained:.4f}")
    return train_p, val_p, test_p, pca


def run_pca(
    splits_dir: str = "splits",
    pca_dim: int = 10,
):
    """
    Compute KMO & Bartlett, run PCA on embeddings,
    save PCA model and PCA embeddings.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(exist_ok=True)

    train, val, test = load_embeddings(splits_path)

    # diagnostics
    kmo_val = kmo(train)
    chi_sq, df, p_val = bartlett(train)
    print(f"KMO={kmo_val:.4f}")
    print(f"Bartlett: chi2={chi_sq:.2f}, df={df:.0f}, p={p_val:.4e}")

    # PCA
    train_p, val_p, test_p, pca = run_pca_on_embeddings(
        train, val, test, n_components=pca_dim
    )

    plot_pca_variance(pca, splits_path)

    # save PCA model + embeddings
    dump(pca, splits_path / "pca_test.joblib")
    np.save(splits_path / "train_embeddings_pca_test.npy", train_p)
    np.save(splits_path / "val_embeddings_pca_test.npy",   val_p)
    np.save(splits_path / "test_embeddings_pca_test.npy",  test_p)

    print("Saved:")
    print(" - pca.joblib")
    print(" - train/val/test_embeddings_pca.npy in", splits_path)


if __name__ == "__main__":
    run_pca()
    
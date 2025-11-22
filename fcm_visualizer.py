import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src import config as cf




def analyze_clusters(memberships: np.ndarray) -> None:
    labels = memberships.argmax(axis=1)
    k = memberships.shape[1]
    n = len(labels)

    print("\nCluster summary")
    for i in range(k):
        mask = labels == i
        count = mask.sum()
        frac = count / n if n else 0.0
        avg_mu = memberships[mask, i].mean() if count else 0.0
        print(f"  {i}: n={count}, frac={frac:.3f}, avg μ={avg_mu:.3f}")


def _to_2d(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 2:
        return x
    return PCA(n_components=2).fit_transform(x)


def plot_2d(
    embeddings: np.ndarray,
    memberships: np.ndarray,
    centers: np.ndarray | None = None,
) -> None:
    z = _to_2d(embeddings)
    labels = memberships.argmax(axis=1)

    plt.figure()
    plt.scatter(z[:, 0], z[:, 1], s=8, c=labels)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.title("FCM clusters")

    if centers is not None:
        c2d = _to_2d(centers)
        plt.scatter(c2d[:, 0], c2d[:, 1], s=80, marker="X")

    plt.tight_layout()
    plt.show()


def plot_memberships(memberships: np.ndarray, max_samples: int = 200) -> None:
    n = min(len(memberships), max_samples)
    plt.figure()
    for i in range(n):
        plt.plot(memberships[i], alpha=0.2)
    plt.xlabel("cluster")
    plt.ylabel("membership")
    plt.title(f"First {n} membership vectors")
    plt.tight_layout()
    plt.show()


def show_top_samples(
    memberships: np.ndarray,
    cluster_id: int,
    n: int = 10,
) -> np.ndarray:
    scores = memberships[:, cluster_id]
    idx = np.argsort(scores)[::-1][:n]

    print(f"\nTop {n} samples for cluster {cluster_id}")
    for rank, i in enumerate(idx, start=1):
        print(f"  #{rank}: idx={i}, μ={scores[i]:.3f}")
    return idx


def visualize_default() -> None:
    emb = np.load(f"{cf.SPLITS_DIR}/train_embeddings_pca.npy")
    mem = np.load(f"{cf.SPLITS_DIR}/train_memberships.npy")
    ctr = np.load(f"{cf.SPLITS_DIR}/fcm_centers.npy")
    analyze_clusters(mem)
    plot_2d(emb, mem, ctr)
    plot_memberships(mem)


if __name__ == "__main__":
    visualize_default()
    # Example: inspect most typical samples of cluster 0
    mem = np.load(f"{cf.SPLITS_DIR}/train_memberships.npy")
    show_top_samples(mem, cluster_id=1, n=5)

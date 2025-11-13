import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def analyze_clusters(memberships):
    """Show cluster distribution and statistics."""
    labels = np.argmax(memberships, axis=1)
    n_clusters = memberships.shape[1]
    
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    for i in range(n_clusters):
        mask = labels == i
        count = mask.sum()
        pct = 100 * count / len(memberships)
        mean_mem = memberships[mask, i].mean()
        
        print(f"\nCluster {i}:")
        print(f"  Samples: {count} ({pct:.1f}%)")
        print(f"  Avg membership: {mean_mem:.3f}")


def show_top_samples(memberships, cluster_id, n=5):
    """Show top N samples for a cluster with their indices."""
    cluster_mem = memberships[:, cluster_id]
    top_idx = np.argsort(cluster_mem)[-n:][::-1]
    
    print(f"\nTop {n} samples in Cluster {cluster_id}:")
    print(f"{'Index':<10} {'Membership'}")
    print("-" * 25)
    for idx in top_idx:
        print(f"{idx:<10} {cluster_mem[idx]:.4f}")
    
    return top_idx


def plot_clusters(embeddings, memberships, centers):
    """Plot clusters in 2D using PCA."""
    # Reduce to 2D
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)
    
    if centers.shape[0] == embeddings.shape[1]:
        ctr_2d = pca.transform(centers.T)
    else:
        ctr_2d = pca.transform(centers)
    
    # Get labels
    labels = np.argmax(memberships, axis=1)
    n_clusters = memberships.shape[1]
    
    # Plot
    plt.figure(figsize=(10, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i in range(n_clusters):
        mask = labels == i
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                   c=colors[i], label=f'Cluster {i}', 
                   alpha=0.6, s=50)
    
    plt.scatter(ctr_2d[:, 0], ctr_2d[:, 1], 
               c='red', marker='X', s=300, 
               edgecolors='black', linewidth=2,
               label='Centers', zorder=10)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.title('Cluster Visualization')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_memberships(memberships):
    """Plot membership distributions."""
    n_clusters = memberships.shape[1]
    
    fig, axes = plt.subplots(1, n_clusters, figsize=(5*n_clusters, 4))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i, ax in enumerate(axes):
        mem = memberships[:, i]
        ax.hist(mem, bins=30, color=f'C{i}', alpha=0.7, edgecolor='black')
        ax.axvline(mem.mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Membership Value')
        ax.set_ylabel('Count')
        ax.set_title(f'Cluster {i} (mean={mem.mean():.3f})')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Quick run everything
def visualize_all(embeddings, memberships, centers):
    """Run all visualizations."""
    analyze_clusters(memberships)
    plot_clusters(embeddings, memberships, centers)
    plot_memberships(memberships)


# Example usage
if __name__ == "__main__":
    # Load data
    emb = np.load("splits/train_embeddings.npy")
    mem = np.load("splits/train_memberships.npy")
    ctr = np.load("splits/fcm_centers.npy")
    
    # Show everything
    visualize_all(emb, mem, ctr)
    
    # Check specific cluster samples
    indices = show_top_samples(mem, cluster_id=0, n=5)
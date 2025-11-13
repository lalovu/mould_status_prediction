# cluster_sequences.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SPLITS = "splits"     # directory with *_windows.npy and *_memberships.npy
SPLIT  = "train"      # "train" | "val" | "test"
CLUSTER_ID = 2       # which cluster to analyze
SENSOR_IDX = 2        # <-- set to TVOC column index
N_EXAMPLES = 12       # examples to show in small-multiples
SAVE_FILES = True     # save idx/X to disk
SAVE_FIGS  = False    # save figures

def load_data(split):
    X   = np.load(f"{SPLITS}/{split}_windows.npy")        # (N, T, F)
    mem = np.load(f"{SPLITS}/{split}_memberships.npy")    # (N, C)
    labels = mem.argmax(axis=1)                           # hard labels
    return X, mem, labels

def save_cluster_arrays(X, labels, k):
    idx = np.where(labels == k)[0]
    if SAVE_FILES:
        out = Path(f"{SPLITS}/{SPLIT}_clusters"); out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"idx_cluster{k}.npy", idx)
        np.save(out / f"X_cluster{k}.npy",   X[idx])
    return idx

def plot_band(matrix, title):
    # matrix: (n_seq, T)
    p10  = np.percentile(matrix, 10, axis=0)
    p90  = np.percentile(matrix, 90, axis=0)
    mean = matrix.mean(axis=0)
    T = np.arange(matrix.shape[1])

    plt.figure(figsize=(9, 4))
    plt.fill_between(T, p10, p90, alpha=0.25, label="10-90% band")
    plt.plot(T, mean, linewidth=2, label="Mean")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Sensor value")
    plt.title(title)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()

def plot_examples(matrix, n, title):
    k = min(n, matrix.shape[0])
    # pick top-membership indices inside this cluster for representativeness
    rows = int(np.ceil(k / 4)); cols = min(4, k)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 2.4*rows), squeeze=False)
    T = np.arange(matrix.shape[1])

    for ax, i in zip(axes.flat, range(k)):
        ax.plot(T, matrix[i], linewidth=1.5)
        ax.set_title(f"seq {i}"); ax.grid(alpha=0.3)
    for ax in axes.flat[k:]:
        ax.axis("off")

    fig.suptitle(title); fig.tight_layout(rect=(0,0,1,0.96))

def main():
    X, mem, labels = load_data(SPLIT)
    idx = save_cluster_arrays(X, labels, CLUSTER_ID)

    if len(idx) == 0:
        print(f"No sequences in cluster {CLUSTER_ID} for split '{SPLIT}'."); return

    tvoc_matrix = X[idx, :, SENSOR_IDX]   # (n_in_cluster, T)

    # plots
    title_band = f"{SPLIT.upper()} - Cluster {CLUSTER_ID} - Sensor {SENSOR_IDX} (mean & 10-90%)"
    plot_band(tvoc_matrix, title_band)

    title_examples = f"{SPLIT.upper()} - Cluster {CLUSTER_ID} - Example sequences"
    plot_examples(tvoc_matrix, N_EXAMPLES, title_examples)

    if SAVE_FIGS:
        figdir = Path(f"{SPLITS}/figs"); figdir.mkdir(exist_ok=True)
        plt.savefig(figdir / f"{SPLIT}_cluster{CLUSTER_ID}_sensor{SENSOR_IDX}.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()

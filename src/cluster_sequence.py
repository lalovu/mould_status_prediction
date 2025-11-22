import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SPLITS = Path("splits")
SPLIT = "train"
SENSOR_IDX = 2  # example: Sensor 2 (TVOC, etc.)

def load_data(split):
    X = np.load(SPLITS / f"{split}_windows.npy")
    mem = np.load(SPLITS / f"{split}_memberships.npy")
    labels = mem.argmax(axis=1)

    # load metadata
    meta = pd.read_csv(SPLITS / f"{split}_metadata.csv")

    return X, labels, meta


def plot_cluster_sequences(X, labels, meta, sensor_idx, split):
    clusters = np.unique(labels)
    T = np.arange(X.shape[1])

    fig, axes = plt.subplots(len(clusters), 1,
                             figsize=(9, 3 * len(clusters)),
                             sharex=True)

    if len(clusters) == 1:
        axes = [axes]

    for ax, k in zip(axes, clusters):
        idx = np.where(labels == k)[0]
        mat = X[idx, :, sensor_idx]

        if mat.size == 0:
            ax.set_title(f"{split.upper()} - Cluster {k} (no data)")
            ax.axis("off")
            continue

        p10 = np.percentile(mat, 10, axis=0)
        p90 = np.percentile(mat, 90, axis=0)
        mean = mat.mean(axis=0)

        ax.fill_between(T, p10, p90, alpha=0.25, label="10–90% band")
        ax.plot(T, mean, linewidth=2, label="Mean")
        ax.set_ylabel("Value")
        ax.set_title(f"{split.upper()} - Cluster {k}")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time (minutes)")
    fig.suptitle(f"{split.upper()} - Sensor {sensor_idx} - All clusters")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()



def interactive_viewer(X, labels, meta, memberships, sensor_idx, split):
    """
    Interactive viewer:
      - arrow keys to navigate
      - shows cluster, membership values, sequence_id, start_time, end_time
    """

    clusters = np.unique(labels)
    T = np.arange(X.shape[1])

    # map: cluster -> indices belonging to that cluster
    cluster_idx = {k: np.where(labels == k)[0] for k in clusters}

    fig, ax = plt.subplots(figsize=(10, 4))
    state = {"cpos": 0, "spos": 0}

    def update_plot():
        ax.clear()

        k = clusters[state["cpos"]]
        idxs = cluster_idx[k]

        if len(idxs) == 0:
            ax.set_title(f"{split.upper()} - Cluster {k} (empty)")
            fig.canvas.draw_idle()
            return

        # clamp
        state["spos"] = max(0, min(state["spos"], len(idxs) - 1))
        seq_idx = idxs[state["spos"]]

        # sequence data
        y = X[seq_idx, :, sensor_idx]

        # metadata
        seq_id = meta.loc[seq_idx, "sequence_id"]
        start_t = meta.loc[seq_idx, "start_time"]
        end_t   = meta.loc[seq_idx, "end_time"]

        # membership vector for this sequence
        mu = memberships[seq_idx]              # shape (3,)
        mu_str = np.array2string(mu, precision=3, separator=", ")

        # plot
        ax.plot(T, y, linewidth=1.5)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Value")

        # updated title — membership instead of Seq x/y
        ax.set_title(
            f"{split.upper()} - Cluster {k} | "
            f"μ = {mu_str} | "
            f"Seq ID: {seq_id} | "
            f"{start_t} → {end_t}"
        )

        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "left":
            state["spos"] -= 1
        elif event.key == "right":
            state["spos"] += 1
        elif event.key == "up":
            state["cpos"] = (state["cpos"] - 1) % len(clusters)
            state["spos"] = 0
        elif event.key == "down":
            state["cpos"] = (state["cpos"] + 1) % len(clusters)
            state["spos"] = 0
        elif event.key in ("q", "escape"):
            plt.close(fig)
            return

        update_plot()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_plot()
    plt.show()




def main():
    X = np.load(SPLITS / f"{SPLIT}_windows.npy")
    memberships = np.load(SPLITS / f"{SPLIT}_memberships.npy")
    labels = memberships.argmax(axis=1)
    meta = pd.read_csv(SPLITS / f"{SPLIT}_metadata.csv")

    plot_cluster_sequences(X, labels, meta, SENSOR_IDX, SPLIT)
    interactive_viewer(X, labels, meta, memberships, SENSOR_IDX, SPLIT)



if __name__ == "__main__":
    main()

import pandas as pd
from . import config as cf
import matplotlib.pyplot as plt
import glob
import os



def count_issues(df):
    # Count
    n_missing = df[cf.DATETIME_COL].isna().sum()
    n_dupes = df.duplicated(subset=[cf.DATETIME_COL]).sum()

    # Previews
    missing_preview = df[df[cf.DATETIME_COL].isna()].head(10)
    dupes_preview = df[df.duplicated(subset=[cf.DATETIME_COL], keep=False)] \
                    .sort_values(cf.DATETIME_COL) \
                    .head(20)

    # report
    print(f"Total records loaded: {len(df)}")
    print(f"Number of missing timestamps: {n_missing}")
    print(f"Number of duplicate timestamps: {n_dupes}")
    print("\nPreview of missing timestamps:")
    print(missing_preview.to_string(index=False))
    print("\nPreview of duplicate timestamps:")
    print(dupes_preview.to_string(index=False))

def reindex_report(df, raw_df):
    print(f"Reindexed data from {df[cf.DATETIME_COL].min()} to {df[cf.DATETIME_COL].max()}")
    print(f"Total records after reindexing: {len(df)}")
    print(f"Preview {df.head(20)}")
    print("-" * 40)


def filter_report(original_df, filtered_df, out_dir):
    # ensures: csv_checklist/<dataset_name>/
    os.makedirs(out_dir, exist_ok=True)

    filtered_df.to_csv(f"{out_dir}/filtered_output.csv", index=False)
    original_df.to_csv(f"{out_dir}/original_output.csv", index=False)


def interpolate_report(original_df, filled_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    filled_df.to_csv(f"{out_dir}/filled_df.csv", index=False)
    original_df.to_csv(f"{out_dir}/original_df.csv", index=False)


def seg_to_csv(segments, out_dir):
    # ensures: csv_checklist/<dataset_name>/segments/
    seg_dir = f"{out_dir}/segments"
    os.makedirs(seg_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        seg.to_csv(f"{seg_dir}/segment_{i}.csv", index=False)


def interactive_plots(states, sensor_cols, datetime_col):

    fig, ax = plt.subplots(figsize=(15, 6))
    state = {"i": 0}  # index of current sensor
    colors = ["blue", "orange", "green", "red", "purple", "cyan"]

    def draw():
        ax.clear()
        sensor = sensor_cols[state["i"]]
        for j, (name, df) in enumerate(states.items()):
            if sensor not in df.columns: 
                continue
            ax.plot(df[datetime_col], df[sensor],
                    label=name, color=colors[j % len(colors)], alpha=0.7)

        ax.set_title(f"{sensor} Data Processing Steps")
        ax.set_xlabel("Timestamp"); ax.set_ylabel(sensor)
        ax.legend(); ax.grid(True, alpha=0.4)
        fig.canvas.draw_idle()

    def on_key(e):
        if e.key == "right":
            state["i"] = (state["i"] + 1) % len(sensor_cols)
            draw()
        elif e.key == "left":
            state["i"] = (state["i"] - 1) % len(sensor_cols)
            draw()

    fig.canvas.mpl_connect("key_release_event", on_key)
    draw()
    plt.show()


def load_segments(path="csv_checklist/segment_*.csv"):
    files = sorted(glob.glob(path))
    if not files:
        raise FileNotFoundError("No CSV files found under csv_checklist/")
    return [pd.read_csv(f) for f in files]


def cop_plot(y, cps, s, false_alarm_indices=None):
    plt.figure(figsize=(15, 5))
    plt.plot(y, alpha=0.7)
    
    # Shade false alarm regions
    if false_alarm_indices:
        for start, end in false_alarm_indices:
            plt.axvspan(start, end, alpha=0.2, color="red")
    
    # Mark change points
    for cp in cps[:-1]:  # skip last one (end of series)
        plt.axvline(cp, color="orange", lw=1, alpha=0.5)
    
    plt.title(f"{s}: Change points")
    plt.tight_layout()
    plt.show()




def save_processed_segments(segment_paths, segments, sensor_cols):
    """
    Save each processed *combined* dataset into:
      processed/<dataset>_dataset/<dataset>_combined_dataset_processed.csv
    """

    for path, df in zip(segment_paths, segments):
        filename = os.path.basename(path)             # e.g. quarry_combined_dataset.csv
        name, _ = os.path.splitext(filename)          # quarry_combined_dataset

        # dataset name: "quarry", "tuba", etc.
        dataset = name.split("_")[0]                  # quarry

        # output directory: processed/quarry_dataset/
        out_dir = os.path.join("processed", f"{dataset}_dataset")
        os.makedirs(out_dir, exist_ok=True)

        # keep timestamp + sensors + FA flags (if any)
        fa_cols = [f"{s}_fa" for s in sensor_cols if f"{s}_fa" in df.columns]
        keep_cols = [cf.DATETIME_COL] + list(sensor_cols) + fa_cols
        df = df[keep_cols]

        # final file path: processed/quarry_dataset/quarry_combined_dataset_processed.csv
        out_path = os.path.join(out_dir, f"{name}_processed.csv")
        df.to_csv(out_path, index=False)

        print(f"Saved {out_path}")















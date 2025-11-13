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


def interpolate_report(original_df, filled_df):
    total_points = len(original_df)
    total_missing = original_df[cf.SENSOR_COLUMNS].isna().sum().sum()
    total_filled = filled_df[cf.SENSOR_COLUMNS].isna().sum().sum()
    total_interpolated = total_missing - total_filled

    filled_df.to_csv('csv_checklist/filled_df.csv', index=False)

    print(filled_df.head(20))
    print(f"Total data points: {total_points * len(cf.SENSOR_COLUMNS)}")
    print(f"Total missing data points before interpolation: {total_missing}")
    print(f"Total data points filled by interpolation: {total_interpolated}")
    print("-" * 40)

def filter_report(original_df, filtered_df):
    filtered_df.to_csv('csv_checklist/filtered_output.csv', index=False)
    original_df.to_csv('csv_checklist/original_output.csv', index=False)

    # Report
    print(f"Total records after filtering: {len(filtered_df)}")
    print(f"Preview of filtered data:")
    print(filtered_df.head(20).to_string(index=False))
    print("-" * 40)

def seg_to_csv(segment):

    os.makedirs("segments", exist_ok=True)

    for i, seg in enumerate(segment):
            filename = f"segments/segment_{i}.csv"
            seg.to_csv(filename, index=False)
            print(f"Saved {filename}")


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













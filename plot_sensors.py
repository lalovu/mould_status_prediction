import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num  
from src import config as cf
import matplotlib.patches as mpatches

# Use full timestamp strings here
CLEAN_PHASES = [
    ("2025-09-29 01:08:20", "2025-10-21 16:12:10")
   
]

def plot_sensors(csv_path: str):
    sensor_cols = list(cf.SENSOR_COLUMNS)  # 7 sensors
    time_col = cf.DATETIME_COL

    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    df_trend = (df[sensor_cols] - df[sensor_cols].mean()) / df[sensor_cols].std()
    df_clean = df_trend.mask(df_trend.abs() > 3).interpolate(limit_direction="both")
    df_smooth = df_clean.rolling(window=100000, min_periods=1).mean()

    x = df[time_col]

    clean_ranges = [
        (pd.to_datetime(start), pd.to_datetime(end))
        for start, end in CLEAN_PHASES
    ]

    groups = [
        sensor_cols[0:2],
        sensor_cols[2:4],
        sensor_cols[4:7],
    ]

    fig = plt.figure(figsize=(12, 8))
    state = {"page": 0}

    def update():
        fig.clf()

#       fig.suptitle("quarry House Dataset", fontsize=16)

        sensors = groups[state["page"]]
        n = len(sensors)

        for i, s in enumerate(sensors, start=1):
            ax = fig.add_subplot(n, 1, i)
            ax.plot(x, df_smooth[s])
            ax.set_title(f"{s} (smoothed trend)")
            ax.set_ylabel("Trend")
            ax.grid(True)

            # shade clean phases with full timestamps
            for start, end in clean_ranges:
                ax.axvspan(
                    float(date2num(start)),
                    float(date2num(end)),
                    color="#fd0000",
                    alpha=0.15
                )
            
            clean_patch = mpatches.Patch(color="#fd0000", alpha=0.15, label='Mould Growth Phase')
            ax.legend(handles=[clean_patch], loc='upper right')


            if i == n:
                ax.set_xlabel(time_col)

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["page"] = (state["page"] + 1) % len(groups)
            update()
        elif event.key == "left":
            state["page"] = (state["page"] - 1) % len(groups)
            update()
        elif event.key in ("q", "escape"):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    update()
    plt.show()


if __name__ == "__main__":
    plot_sensors("processed/quarry_dataset/quarry_combined_dataset_processed.csv")

# sequence.py
import numpy as np
import pandas as pd
from pathlib import Path
from config import WINDOW, SENSOR_COLUMNS, DATETIME_COL, LARGE_GAP


def slice_windows(df, window_size):
    """Return (n_windows, window_size, n_sensors) or None if too short."""
    arr = df[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    n_full = len(arr) // window_size
    if n_full == 0:
        return None
    n = n_full * window_size
    return arr[:n].reshape(n_full, window_size, arr.shape[1])


def main():
    processed_root = Path("processed")
    out_dir = Path("splits")
    out_dir.mkdir(exist_ok=True)

    split_windows = {"train": [], "val": [], "test": []}
    split_meta    = {"train": [], "val": [], "test": []}
    seq_id = 0


    # processed/<dataset>_dataset/*_processed.csv
    files = sorted(processed_root.glob("*_dataset/*_processed.csv"))
    if not files:
        raise FileNotFoundError("No *_processed.csv found under processed/*_dataset/")

    for file_path in files:
        df = pd.read_csv(file_path, parse_dates=[DATETIME_COL])
        df = df.sort_values(DATETIME_COL).reset_index(drop=True)

        windows = slice_windows(df, WINDOW)
        if windows is None:
            print(f"Skipping {file_path} (not enough rows)")
            continue

        n = windows.shape[0]
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)

        idx_ranges = {
            "train": (0, n_train),
            "val":   (n_train, n_train + n_val),
            "test":  (n_train + n_val, n),
        }

        location = file_path.parent.name
        print(
            f"{file_path} -> "
            f"train {idx_ranges['train'][1] - idx_ranges['train'][0]}, "
            f"val {idx_ranges['val'][1] - idx_ranges['val'][0]}, "
            f"test {idx_ranges['test'][1] - idx_ranges['test'][0]}"
        )

        for split_name, (w0, w1) in idx_ranges.items():
            for w_idx in range(w0, w1):

                row_start = w_idx * WINDOW
                row_end   = row_start + WINDOW - 1

                start_time = df[DATETIME_COL].iloc[row_start]
                end_time   = df[DATETIME_COL].iloc[row_end]


                
                    # timestamp continuity logic
               
                times = df[DATETIME_COL].iloc[row_start:row_end+1].to_numpy()
                diff_sec = np.diff(times).astype("timedelta64[s]").astype(int)

                if np.any(diff_sec > LARGE_GAP):
                    continue  # large gap → drop window

                if np.any(diff_sec != 10):
                    # small gaps → interpolate window
                    win_df = df.iloc[row_start:row_end+1].copy()
                    for s in SENSOR_COLUMNS:
                        win_df[s] = win_df[s].interpolate(
                            method="linear",
                            limit_direction="both"
                        )
                    window = win_df[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
                else:
                    window = df[SENSOR_COLUMNS].iloc[row_start:row_end+1] \
                                .to_numpy(dtype=np.float32)
                # ---------------------------------------

                split_windows[split_name].append(window)
                split_meta[split_name].append({
                    "sequence_id": seq_id,
                    "location": location,
                    "start_time": start_time,
                    "end_time": end_time,
                })
                seq_id += 1

    # Save all splits
 
    for split_name in ["train", "val", "test"]:
        if not split_windows[split_name]:
            print(f"No {split_name} windows found.")
            continue

        arr = np.stack(split_windows[split_name], axis=0)
        meta_df = pd.DataFrame(split_meta[split_name])

        print(f"\n--- Checking NaNs for {split_name} ---")
        print("Total NaNs:", np.isnan(arr).sum())
        print("----------------------------------\n")

        np.save(out_dir / f"{split_name}_windows.npy", arr)
        meta_df.to_csv(out_dir / f"{split_name}_metadata.csv", index=False)

        print(f"{split_name}: {arr.shape[0]} sequences saved")


if __name__ == "__main__":
    main()

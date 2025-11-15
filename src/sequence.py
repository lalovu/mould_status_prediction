import numpy as np
import pandas as pd
from pathlib import Path
from config import WINDOW, SENSOR_COLUMNS


def slice_windows(df, window_size):
    arr = df[SENSOR_COLUMNS].to_numpy(dtype=np.float32)
    n_full = len(arr) // window_size
    if n_full == 0:
        return None
    n = n_full * window_size
    return arr[:n].reshape(n_full, window_size, arr.shape[1])


def main():
  
    paths = sorted(Path("processed").glob("*/*_combined_dataset_processed.csv"))
    if not paths:
        raise FileNotFoundError("No *_combined_dataset_processed.csv found under processed/*/")

    split_windows = {"train": [], "val": [], "test": []}
    split_meta = {"train": [], "val": [], "test": []}
    seq_id = 0

    for p in paths:
        df = pd.read_csv(p)
        windows = slice_windows(df, WINDOW)
        if windows is None:
            print(f"Skipping {p} (not enough rows)")
            continue

        location = p.parent.name
        n = windows.shape[0]

        # per-file 70/15/15 split (chronological within this file)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        idx_train = (0, n_train)
        idx_val = (n_train, n_train + n_val)
        idx_test = (n_train + n_val, n)

        ranges = {
            "train": idx_train,
            "val": idx_val,
            "test": idx_test,
        }

        print(
            f"{p} -> train {idx_train[1] - idx_train[0]}, "
            f"val {idx_val[1] - idx_val[0]}, "
            f"test {idx_test[1] - idx_test[0]}"
        )

        for split_name, (i0, i1) in ranges.items():
            if i1 <= i0:
                continue  # nothing in this split for this file

            # windows for this split and this file
            split_windows[split_name].append(windows[i0:i1])

            # metadata aligned with these windows
            for _ in range(i1 - i0):
                split_meta[split_name].append(
                    {
                        "sequence_id": seq_id,
                        "location": location,
                    }
                )
                seq_id += 1

                

    out_dir = Path("splits")
    out_dir.mkdir(exist_ok=True)

    for split_name in ["train", "val", "test"]:
        if not split_windows[split_name]:
            print(f"No windows for {split_name}, skipping.")
            continue

        arr = np.concatenate(split_windows[split_name], axis=0)
        meta_df = pd.DataFrame(split_meta[split_name]).reset_index(drop=True)

        print(f"\n--- Checking NaNs for {split_name} windows ---")
        print("Total NaNs:", np.isnan(arr).sum())
        print("-------------------------------------------\n")

        np.save(out_dir / f"{split_name}_windows.npy", arr)
        meta_df.to_csv(out_dir / f"{split_name}_metadata.csv", index=False)

        print(f"{split_name}: {arr.shape[0]} sequences saved")

    


if __name__ == "__main__":
    main()

  


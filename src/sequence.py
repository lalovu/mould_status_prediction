import pandas as pd
import numpy as np
from pathlib import Path
import config as cf

# 1) Lock the exact feature order ONCE and reuse everywhere
SENSOR_COLS = cf.SENSOR_COLUMNS
def slice_windows(df, window_size=360):
    # 2) Use ONLY sensor columns; ensure numeric and correct order
    arr = df[SENSOR_COLS].to_numpy(dtype=np.float32)
    n = len(arr) // window_size * window_size
    return arr[:n].reshape(-1, window_size, arr.shape[1])

segments = Path('processed').glob('processed_segment_*.csv')
all_windows = []
start_times = []

WINDOW = 360  # avoid hardcoding 360 elsewhere

print("="*60)
print(f"{'Segment':<30} {'Rows':<10} {'Windows':<10} {'Hours':<10}")
print("="*60)

for file in sorted(segments):
    df = pd.read_csv(file, parse_dates=['timestamp'])
    df = df.sort_values('timestamp')

    windows = slice_windows(df, window_size=WINDOW)

    n_w = len(windows)
    if n_w > 0:
        idx = np.arange(n_w) * WINDOW
        start_times.extend(df['timestamp'].iloc[idx].to_list())

    print(f"{file.name:<30} {len(df):<10,} {n_w:<10} {float(n_w):<10.2f}")
    # extend with (window_size, num_features) arrays
    all_windows.extend(windows)

print("="*60)
print(f"{'TOTAL':<30} {'':<10} {len(all_windows):<10} {float(len(all_windows)):<10.2f}")
print("="*60)

if len(all_windows) != len(start_times):
    raise ValueError("Window count and start_times count mismatch.")

# 3) Chronological sort across segments
start_times = np.array(start_times)
order = np.argsort(start_times)
all_windows = [all_windows[i] for i in order]

# 4) Chronological split 70/15/15
N = len(all_windows)
n_train = int(np.floor(0.70 * N))
n_val   = int(np.floor(0.15 * N))
n_test  = N - n_train - n_val

train_windows = np.stack(all_windows[:n_train], axis=0)                  # (N_tr, 360, 7)
val_windows   = np.stack(all_windows[n_train:n_train+n_val], axis=0)     # (N_va, 360, 7)
test_windows  = np.stack(all_windows[n_train+n_val:], axis=0)            # (N_te, 360, 7)

print(f"Split -> Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}")

Path('splits').mkdir(exist_ok=True)
np.save('splits/train_windows.npy', train_windows)
np.save('splits/val_windows.npy',   val_windows)
np.save('splits/test_windows.npy',  test_windows)



print("Saved: splits/train_windows.npy, splits/val_windows.npy, splits/test_windows.npy")

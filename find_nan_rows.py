import pandas as pd
from pathlib import Path
from src.config import SENSOR_COLUMNS

paths = sorted(Path("processed").glob("*_dataset/*_processed.csv"))

for p in paths:
    df = pd.read_csv(p)

    # rows where ANY sensor column is NaN
    nan_mask = df[SENSOR_COLUMNS].isna().any(axis=1)
    nan_rows = df[nan_mask]

    if nan_rows.empty:
        print(f"{p} - No NaNs found.")
    else:
        print(f"\n=== NaNs FOUND in: {p} ===")
        print(f"Total NaN rows: {len(nan_rows)}")
        print("Row indices:", nan_rows.index.tolist())  # exact row numbers

        # Show the problematic rows
        print("\nRows with NaNs:")
        print(nan_rows)

        # Show which columns specifically have NaN
        print("\nColumns with NaN per row:")
        print(nan_rows.isna().sum(axis=1))

# merge_daily_logs.py

from pathlib import Path
import pandas as pd
from src import config as cf  

def merge_logs(input_folder, output_folder, output_name="all_logs_merged.csv"):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("log_*.csv"))
    if not files:
        raise FileNotFoundError(f"No log_*.csv files found in {input_folder}")

    dfs = []
    for f in files:
        print(f"Reading {f.name} ...")
        dfs.append(pd.read_csv(f))

    big = pd.concat(dfs, ignore_index=True)

    # --- rename columns (only added step)
    rename_map = {
        "Timestamp": cf.DATETIME_COL,
        "Temp(C)": "temp",
        "Humidity(%)": "humidity",
        "TVOC(ppb)": "tvoc",
        "eCO2(ppm)": "eCO2",
        "PM1": "pm1",
        "PM2.5": "pm2.5",
        "PM10": "pm10",
    }
    big = big.rename(columns=rename_map)

    # --- chronological sorting (same as your original)
    big[cf.DATETIME_COL] = pd.to_datetime(
        big[cf.DATETIME_COL],
        format="%Y-%m-%d %H-%M-%S",
        errors="coerce"
    )
    big = big.sort_values(cf.DATETIME_COL).reset_index(drop=True)

    out_path = output_folder / output_name
    big.to_csv(out_path, index=False)

    print(f"\nSaved merged dataset to: {out_path}  Shape={big.shape}")
    return big


# Example usage exactly as you wrote
merge_logs(
    input_folder="dataset/quarry_data",
    output_folder="dataset",
    output_name="quarry_combined_dataset.csv"
)

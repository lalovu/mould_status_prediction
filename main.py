from src import config as cf
from src import preprocess as pre

def main():
    for path in cf.DATA_PATHS:
        raw, nat_count, duplicate_count = pre.load_sensor_data(path)
        print(f"Loaded {path}: {raw.shape[0]} rows, {nat_count} NaT in datetime, {duplicate_count} duplicate timestamps")




if __name__ == "__main__":
    main()
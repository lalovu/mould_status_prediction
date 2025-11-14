from src import config as cf
from src import preprocess as pre
from src import utils as ut
from pathlib import Path


def main():
    eval = cf.EVAL  # kept for now in case you use it elsewhere

    for path in cf.DATA_PATHS:
        path = Path(path)
        dataset_name = path.stem
        out_dir = Path(cf.OUTPUT_ROOT) / dataset_name

        print(f"\n=== Processing dataset: {dataset_name} ===")

        # Load data
        raw, sensor, df = pre.load_sensor_data(path)

        # Reindex to continuous time grid
        rd = pre.reindex(raw)

        # Hampel filter (outlier detection)
        hampel_df = pre.hampel_filter(rd)

        # Interpolate small gaps
        fill_seg = pre.interpolate(hampel_df)

        # Segment continuous regions (no large gaps)
        segments, report = pre.segment_gaps(fill_seg)

        # Apply rolling median smoothing per segment
        processed_segments = [pre.rolling(seg) for seg in segments]

        # Ensure output directory exists
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reports & CSV exports
        ut.filter_report(rd, hampel_df, out_dir)
        ut.interpolate_report(rd, fill_seg, out_dir)
        ut.seg_to_csv(processed_segments, out_dir)

        # Interactive plots (no folder needed)
        states = {
            "raw": raw,
            "hampel": hampel_df,
        }
        ut.interactive_plots(
            states,
            sensor_cols=cf.SENSOR_COLUMNS,
            datetime_col=cf.DATETIME_COL,
        )


if __name__ == "__main__":
    main()

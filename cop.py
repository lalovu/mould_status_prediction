# cop.py
import numpy as np
import ruptures as rpt
from src import config as cf
from src import utils as ut
import glob

eval = cf.EVAL


def false_alarm_event(df, segment_df, y, cps, sensor, threshold):
    """
    Handle false alarm detection per sensor.
    Short blocks -> interpolate.
    Long blocks (length > cf.MAX_FA_GAP) -> mark to drop later.
    """
    results = []

    # ensure cps are valid boundaries
    cps = [0] + [c for c in cps if 0 < c < len(y)] + [len(y)]

    # false alarm flag for this sensor
    fa_col = f"{sensor}_fa"
    if fa_col not in segment_df.columns:
        segment_df[fa_col] = False

    # global drop mask (merged across sensors)
    if "drop_row" not in segment_df.columns:
        segment_df["drop_row"] = False

    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        segment = y[start:end]

        # if false alarm present
        if (segment > threshold).any():
            results.append(df.iloc[start:end][[sensor]])

            seg_len = end - start

            # mark FA region
            segment_df.loc[start:end - 1, fa_col] = True

            if seg_len <= cf.MAX_FA_GAP:
                # short -> interpolate later
                segment_df.loc[start:end - 1, sensor] = np.nan
            else:
                # long -> drop these rows entirely later
                segment_df.loc[start:end - 1, "drop_row"] = True

    # interpolate NaN created by short blocks only
    if segment_df[sensor].isna().any():
        segment_df[sensor] = segment_df[sensor].interpolate(
            method="linear", limit_direction="both"
        )

    return results, segment_df



def cop_false_alarm(segments, return_segments=False, segment_names=None):
    """
    Perform CoP (change point) detection and false alarm removal for each segment.
    """
    all_false_alarms = []
    processed_segments = []

    for idx, df in enumerate(segments):
        print(f"\n=== Processing Segment {idx + 1} ===")
        seg_name = segment_names[idx] if segment_names else f"Segment {idx+1}"
        print(f"File: {seg_name}")

        segment_df = df.copy()

        for sensor in cf.SENSOR_COLUMNS:
            y = segment_df[sensor].to_numpy(dtype=float)

            # standardization
            med = np.nanmedian(y)
            mad = 1.4826 * np.nanmedian(np.abs(y - med))
            y_std = (y - med) / (mad if mad > 0 else 1)

            # CoP detection (PELT)
            try:
                algo = rpt.Pelt(
                    model=cf.MODEL, min_size=cf.MINL, jump=cf.JUMP
                )
                cps = algo.fit(y_std).predict(pen=cf.PEN)
            except Exception:
                # if segment is too short or unstable for PELT
                cps = [len(y)]

            # per-sensor threshold
            thr = cf.FALSE_ALARM_THRESHOLDS.get(sensor, 50)

            # false alarm logic
            fa_list, segment_df = false_alarm_event(
                df=df,
                segment_df=segment_df,
                y=y_std,
                cps=cps,
                sensor=sensor,
                threshold=thr,
            )
            all_false_alarms.extend(fa_list)

        # ---- DROP ROWS FROM LONG FALSE-ALARM BLOCKS ----
        if "drop_row" in segment_df.columns:
            before = len(segment_df)
            segment_df = segment_df.loc[~segment_df["drop_row"]].reset_index(drop=True)
            dropped = before - len(segment_df)
            if dropped > 0:
                print(f"Dropped {dropped} rows (long false alarms).")
            segment_df = segment_df.drop(columns=["drop_row"])

        processed_segments.append(segment_df)

    if return_segments:
        return all_false_alarms, processed_segments
    return all_false_alarms



if __name__ == "__main__":
    # 1) Get segment file paths 
    segment_files = sorted(glob.glob("dataset/*_dataset.csv"))
    if not segment_files:
        raise FileNotFoundError("No CSV files found under dataset/_dataset.csv/")

    # 2) Load the segments
    segments = ut.load_segments("dataset/*_dataset.csv")

    # 3) CoP + false alarm detection
    all_false_alarms, outputs = cop_false_alarm(
        segments,
        return_segments=True,
        segment_names=segment_files,  
    )

    print(
        f"Processed {len(outputs)} segments with "
        f"{len(all_false_alarms)} false alarm events"
    )
    
    ut.save_processed_segments(
        segment_paths=segment_files,
        segments=outputs,
        sensor_cols=cf.SENSOR_COLUMNS,
    )
import os
import glob
import numpy as np
import ruptures as rpt
import config as cf
import utils as ut

eval = cf.EVAL

def cop_false_alarm(segments, return_segments: bool = False, segment_names=None):

    all_false_alarms = []
    processed_segments = []

    for idx, df in enumerate(segments):
        # Print the actual file name 
        if segment_names is not None and idx < len(segment_names):
            seg_path = segment_names[idx]
            seg_name = os.path.basename(seg_path)
            print(f"\n=== Processing {seg_name} ===")
        else:
            print(f"\n=== Processing Segment {idx + 1} ===")

        segment_df = df.copy()

        for sensor in cf.SENSOR_COLUMNS:
            # --- Standardization 
            y = df[sensor].to_numpy(dtype=float)
            med = np.nanmedian(y)
            mad = 1.4826 * np.nanmedian(np.abs(y - med))
            y_std = (y - med) / (mad if mad > 0 else 1)

            # --- Change Point Detection
            algo = rpt.Pelt(model=cf.MODEL, min_size=cf.MINL, jump=cf.JUMP)
            cps = algo.fit(y_std).predict(pen=cf.PEN)

            # --- Per-sensor threshold 
            thr = getattr(cf, "FALSE_ALARM_THRESHOLDS", {}).get(sensor, 50)

            # --- Detect, mark, and interpolate false alarms for this sensor
            false_alarms, segment_df = false_alarm_event(
                df=df,
                segment_df=segment_df,
                y=y_std,
                cps=cps,
                sensor=sensor,
                threshold=thr,
            )
            all_false_alarms.extend(false_alarms)

            if eval:
                # Extract indices from false_alarms for plotting
                fa_indices = [(fa.index[0], fa.index[-1] + 1) for fa in false_alarms]
                ut.cop_plot(y_std, cps, sensor, false_alarm_indices=fa_indices)

        processed_segments.append(segment_df)

    return (all_false_alarms, processed_segments) if return_segments else all_false_alarms


def false_alarm_event(df, segment_df, y, cps, sensor, threshold):
    results = []

    # Ensure boundaries are within series
    cps = [0] + [c for c in cps if 0 < c < len(y)] + [len(y)]

    # Add flag column for this sensor
    flag_col = f"{sensor}_fa"
    if flag_col not in segment_df.columns:
        segment_df[flag_col] = False

    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        segment = y[start:end]

        if (segment > threshold).any():
            # Store false alarm data 
            false_alarm = df.iloc[start:end][[sensor]]
            results.append(false_alarm)

            # Mark this region as false alarm
            segment_df.loc[start:end - 1, flag_col] = True

            # Set this sensor to NaN in this region
            segment_df.loc[start:end - 1, sensor] = np.nan

    if results:
        # Interpolate all NaN values for this sensor at once
        segment_df[sensor] = segment_df[sensor].interpolate(
            method="linear",
            limit_direction="both",
        )

    return results, segment_df


if __name__ == "__main__":
    # 1) Get segment file paths 
    segment_files = sorted(glob.glob("dataset/*_combined_dataset.csv"))
    if not segment_files:
        raise FileNotFoundError("No CSV files found under csv_checklist/")

    # 2) Load the segments
    segments = ut.load_segments("dataset/*_combined_dataset.csv")

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
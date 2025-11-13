import numpy as np
import ruptures as rpt
from src import config as cf
from src import utils as ut
import os

eval = cf.EVAL


def cop_false_alarm(segments, return_segments: bool = False):
    all_false_alarms = []
    processed_segments = []
    
    for idx, df in enumerate(segments):
        print(f"\n=== Processing Segment {idx + 1} ===")
        
        segment_df = df.copy()
        
        for sensor in cf.SENSOR_COLUMNS:
            # Standardize sensor data
            y = df[sensor].to_numpy(dtype=float)
            med = np.nanmedian(y)
            mad = 1.4826 * np.nanmedian(np.abs(y - med))
            y_std = (y - med) / (mad if mad > 0 else 1)
            
            # Detect change points
            algo = rpt.Pelt(model=cf.MODEL, min_size=cf.MINL, jump=cf.JUMP)
            cps = algo.fit(y_std).predict(pen=cf.PEN)
            
            # Detect, mark, and interpolate false alarms for this sensor
            false_alarms, segment_df = false_alarm_event(
                df, segment_df, y_std, cps, sensor, threshold=50
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
    cps = [0] + [c for c in cps if 0 < c < len(y)] + [len(y)]
    
    # Add flag column for this sensor
    flag_col = f"{sensor}_fa"
    segment_df[flag_col] = False
    
    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        segment = y[start:end]
        
        if (segment > threshold).any():
            # Store false alarm data (only timestamp + this specific sensor)
            false_alarm = df.iloc[start:end][[sensor]]
            results.append(false_alarm)
            
            # Mark this region as false alarm
            segment_df.loc[start:end-1, flag_col] = True
            
            # Interpolate ONLY this sensor in this region
            segment_df.loc[start:end-1, sensor] = np.nan
    
    # Interpolate all NaN values for this sensor at once
    if results:  # Only if there were false alarms
        segment_df[sensor] = segment_df[sensor].interpolate(
            method="spline", order = 2, limit_direction="both"
        )
    
    return results, segment_df


# Main execution
segments = ut.load_segments("csv_checklist/segment_*.csv")
all_false_alarms, outputs = cop_false_alarm(segments, return_segments=True)
print(f"Processed {len(outputs)} segments with {len(all_false_alarms)} false alarm events")


os.makedirs("processed", exist_ok=True)


for idx, segment_df in enumerate(outputs):
    # Keep only timestamp, sensor columns, and fa marks
    fa_cols = [f"{s}_fa" for s in cf.SENSOR_COLUMNS]
    keep_cols = ["timestamp"] + list(cf.SENSOR_COLUMNS) + fa_cols
    segment_df = segment_df[keep_cols]

    filename = f"processed/processed_segment_{idx + 1}.csv"
    segment_df.to_csv(filename, index=False)
    print(f"Saved {filename}")

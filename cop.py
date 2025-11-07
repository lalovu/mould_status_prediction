
import numpy as np
import ruptures as rpt
from src import config as cf
from src import utils as ut

eval = cf.EVAL

def cop_false_alarm(segments):
    all_false_alarms = []
    for idx, df in enumerate(segments):
        print(f"\n=== Processing Segment {idx + 1} ===")
        
        for s in cf.SENSOR_COLUMNS:
            y = df[s].to_numpy(dtype=float)

            # 2. STANDARDIZATION 
            med = np.nanmedian(y)
            mad = 1.4826 * np.nanmedian(np.abs(y - med))
            y = (y - med) / (mad if mad > 0 else 1)

            # 3. CHANGE POINT DETECTION
            cps = rpt.Pelt(model=cf.MODEL, min_size=cf.MINL, jump=cf.JUMP).fit(y).predict(pen=cf.PEN)
            
            # 4. MARKING OF FALSE ALARM USING COP
            false_alarms, fa_indices = mark_false_alarms(df, y, cps, threshold=50)
            if eval == True:
                ut.cop_plot(y, cps, s, false_alarm_indices=fa_indices)
            
            all_false_alarms.extend(false_alarms)

            for fa in false_alarms:
                print(f"Sensor: {s}")
                print(fa)
    
    return all_false_alarms

def mark_false_alarms(df, y, cps, threshold=100):
    results = []
    false_alarm_indices = []
    cps = [0] + [c for c in cps if 0 < c < len(y)] + [len(y)]

    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        segment = y[start:end]

        # Trigger event if any value in segment is above threshold
        if (segment > threshold).any():
            # Find the index where y first exceeds threshold
            threshold_indices = np.where(segment > threshold)[0]
            actual_start = start + threshold_indices[0]
            
            false_alarm = df.iloc[actual_start:end][['timestamp'] + cf.SENSOR_COLUMNS]
            results.append(false_alarm)
            false_alarm_indices.append((actual_start, end))

    return results, false_alarm_indices



# Usage:
segments = ut.load_segments("csv_checklist/segment_*.csv")
all_false_alarms = cop_false_alarm(segments)

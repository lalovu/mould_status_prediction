import pandas as pd
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from src import config as cf
from src import utils as ut

# 1. LOAD
df = pd.read_csv("csv_checklist/segment_1.csv")
y = df[cf.SENSOR].to_numpy(dtype=float)

# 2. STANDARDIZATION 
med = np.nanmedian(y)
mad = 1.4826 * np.nanmedian(np.abs(y - med))
y = (y - med) / (mad if mad > 0 else 1)

# 3. CHANGE POINT DETECTION
cps = rpt.Pelt(model=cf.MODEL, min_size=cf.MINL, jump=cf.JUMP).fit(y).predict(pen=cf.PEN)
ut.cop_plot(y, cps)


# 4. MARKING OF FALSE ALARM USING COP
def mark_false_alarms(df, y, cps, threshold=100):
    results = []
    cps = [0] + [c for c in cps if 0 < c < len(y)] + [len(y)]

    for i in range(len(cps) - 1):
        start, end = cps[i], cps[i + 1]
        segment = y[start:end]

        if (segment > threshold).any():
            false_alarm = df.iloc[start:end][['timestamp', 'tvoc']]
            results.append(false_alarm)

    return results

false_alarms = mark_false_alarms(df, y, cps, threshold=100)


for x in false_alarms:
    print(x)







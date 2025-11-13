import pandas as pd
from . import config as cf
import numpy as np
from hampel import hampel


def load_sensor_data(path):
    df = pd.read_csv(
        path,
        usecols = [cf.DATETIME_COL] + cf.SENSOR_COLUMNS,
        encoding = 'utf-8',
        header=0,
    )

    df[cf.DATETIME_COL] = pd.to_datetime(
        df[cf.DATETIME_COL],
        format = cf.FORMAT,
        errors = 'coerce'
    )

    df_drop = df.drop_duplicates(subset=[cf.DATETIME_COL], keep='first')
    df_sensors = df_drop[cf.SENSOR_COLUMNS].apply(pd.to_numeric, errors='coerce')
    
    return df_drop, df_sensors, df


def reindex(df):
    df = df.copy()
    # Beginning and End of Data Gathered
    start = df[cf.DATETIME_COL].min()
    end = df[cf.DATETIME_COL].max()

    full_range = pd.date_range(start=start, end=end, freq=cf.FREQ)
    df_reindexed = df.set_index(cf.DATETIME_COL) \
                    .reindex(full_range) \
                    .rename_axis(cf.DATETIME_COL) \
                    .reset_index()

    return df_reindexed

def hampel_filter(df):
    df_copy = df.copy()
    for col in cf.SENSOR_COLUMNS:
        series = df_copy[col]
        res = hampel(
            series, 
            window_size=cf.HAMPEL_WINDOW, 
            n_sigma=float(cf.HAMPEL_N_SIGMAS)
        )
        
        flag_col = col + "_is_outlier"
        df_copy[flag_col] = 0
        df_copy.loc[res.outlier_indices, flag_col] = 1

    return df_copy   


def interpolate(df):
    df_interp = df.copy()
    df_interp[cf.DATETIME_COL] = pd.to_datetime(df_interp[cf.DATETIME_COL], errors='coerce')
    df_interp = df_interp.set_index(cf.DATETIME_COL)
    
    for col in cf.SENSOR_COLUMNS:
        orig_nan_mask = df_interp[col].isna()
        df_interp[col] = df_interp[col].interpolate(
            method='linear',
            limit=cf.GAP_THRESHOLD,  
            limit_direction='both'
        )
        flag_col = col + "_is_interpolated"
        df_interp[flag_col] = orig_nan_mask & df_interp[col].notna()

    df_interp = df_interp.reset_index()

    return df_interp

def segment_gaps(df):
    df_copy = df.copy()
    df_copy[cf.DATETIME_COL] = pd.to_datetime(df_copy[cf.DATETIME_COL], errors='coerce')
    df_copy = df_copy.set_index(cf.DATETIME_COL)
    
    ok = df_copy[cf.SENSOR_COLUMNS].notna().all(axis=1)

    start = df_copy.index[ok & ~ok.shift(1, fill_value=False)]
    end = df_copy.index[ok & ~ok.shift(-1, fill_value=False)]

    seg = pd.DataFrame({"start": start, "end": end})
    seg["duration"] = seg["end"] - seg["start"]
    segments = [df_copy.loc[s:e].reset_index() for s, e in zip(start, end)]

    df_copy = df_copy.reset_index()

    return segments, seg

def rolling(df):
    df = df.copy()
    df[cf.DATETIME_COL] = pd.to_datetime(df[cf.DATETIME_COL], errors='coerce')
    df = df.set_index(cf.DATETIME_COL)

    for col in cf.SENSOR_COLUMNS:
        series = df[col]
        df[col] = series.rolling(window=30, center=True, min_periods=1).median()
    
    df = df.reset_index()
    return df










    
import pandas as pd
import config as cf
import numpy as np




def load_sensor_data(path):
    df = pd.read_csv(
        path,
        usecols = [cf.DATETIME_COL] + cf.SENSOR_COLUMNS,
        encoding = 'utf-8',
    )

    df[cf.DATETIME_COL] = pd.to_datetime(
        df[cf.DATETIME_COL],
        format = cf.FORMAT,
        errors = 'coerce'
    )

    nat_count = df[cf.DATETIME_COL].isna().sum()
    duplicate_count = df.duplicated(subset=[cf.DATETIME_COL], keep='first').sum()

    df = df.drop_duplicates(subset=[cf.DATETIME_COL], keep='first')

    df_sensors = df[cf.SENSOR_COLUMNS].apply(pd.to_numeric, errors='coerce')
    

    return df_sensors, nat_count, duplicate_count

    return df
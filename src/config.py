
# Column DEFINITION
DATETIME_COL = 'timestamp'
SENSOR_COLUMNS = ['temp', 'humidity', 'tvoc', 'eCO2', 'pm1', 'pm2.5', 'pm10']


# DATASETS
DATA_PATHS = {
    'dataset/tuba_combined_datasets.csv'
}

FORMAT = '%Y-%m-%d %H:%M:%S'

FREQ = '10s'
RESAMP = '5min'
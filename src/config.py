
# Column DEFINITION
DATETIME_COL = 'timestamp'
SENSOR_COLUMNS = ['temp', 'humidity', 'tvoc', 'eCO2', 'pm1', 'pm2.5', 'pm10']


# DATASETS
DATA_PATHS = {
    'dataset/tuba_combined_dataset.csv'
}

FORMAT = '%Y-%m-%d %H:%M:%S'

FREQ = '10s'
RESAMP = '5min'


EVAL = True

GAP_THRESHOLD = int('6')


HAMPEL_WINDOW = 5
HAMPEL_N_SIGMAS = 3
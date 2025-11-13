
# Column DEFINITION
DATETIME_COL = 'timestamp'
SENSOR_COLUMNS = ['temp', 'humidity', 'tvoc', 'eCO2', 'pm1', 'pm2.5', 'pm10']


# DATASETS
DATA_PATHS = {
    'dataset/tuba_combined_dataset.csv'
    'dataset/quarry_combined_dataset.csv'
}

# Preprocessing 
FORMAT = '%Y-%m-%d %H:%M:%S'
FREQ = '10s'
RESAMP = '5min'

GAP_THRESHOLD = int('30')

# Utility enable

EVAL = True

# Filter 

HAMPEL_WINDOW = 5
HAMPEL_N_SIGMAS = 2

# CoP DETECTION

MODEL = 'l2'
PEN = 35000
MINL = 540
JUMP = 360

# 

FA_THRESH = 200

'''

MODEL = 'l2'
PEN = 1000
MINL = 360
JUMP = 1200

'''
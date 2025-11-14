# Column DEFINITION
DATETIME_COL = 'timestamp'
SENSOR_COLUMNS = ['temp', 'humidity', 'tvoc', 'eCO2', 'pm1', 'pm2.5', 'pm10']

# DATASETS
DATA_PATHS = [
    'dataset/tuba_combined_dataset.csv',
    'dataset/quarry_combined_dataset.csv'
]

# Dataset Output Root
OUTPUT_ROOT = 'csv_checklist'
PROCESSED_ROOT = "processed"

# Per-sensor false alarm thresholds 
FALSE_ALARM_THRESHOLDS = {
    'temp': 100,
    'humidity': 100,
    'tvoc': 10,
    'eCO2': 10,
    'pm1': 80,
    'pm2.5': 80,
    'pm10': 80,
}

# Preprocessing 
FORMAT = '%Y-%m-%d %H:%M:%S'
FREQ = '10s'
RESAMP = '5min'

GAP_THRESHOLD = int('30')

# Utility 
EVAL = True

# Filter 
HAMPEL_WINDOW = 5
HAMPEL_N_SIGMAS = 2

# CoP DETECTION

MODEL = 'l2'
PEN = 100
MINL = 360
JUMP = 540

# 

'''

MODEL = 'l2'
PEN = 1000
MINL = 360
JUMP = 1200

'''

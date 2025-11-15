

#------------Utility------------------
EVAL = True

#------------Column Data------------------
DATETIME_COL = 'timestamp'
SENSOR_COLUMNS = ['temp', 'humidity', 'tvoc', 'eCO2', 'pm1', 'pm2.5', 'pm10']

#------------Datasets------------------
DATA_PATHS = [
    'dataset/tuba_combined_dataset.csv',
    'dataset/quarry_combined_dataset.csv'
]

#------------ROOT FOLDER FOR OUTPUTS AND INPUTS------------------
OUTPUT_ROOT = 'csv_checklist'
PROCESSED_ROOT = "processed"

#------------Change of Point False Alarm Event Detection------------------
FALSE_ALARM_THRESHOLDS = {
    'temp': 100,
    'humidity': 100,
    'tvoc': 10,
    'eCO2': 10,
    'pm1': 80,
    'pm2.5': 80,
    'pm10': 80,
}

#------------Change of Point Parameters------------------
MODEL = 'l2'
PEN = 100
MINL = 36
JUMP = 540

MAX_FA_GAP = 60 
LARGE_GAP = 30  # seconds allowed (≤ 3 missing points for 10-sec sampling)

#------------FORMATS------------------
FORMAT = '%Y-%m-%d %H:%M:%S'
FREQ = '10s'
RESAMP = '5min'
GAP_THRESHOLD = int('30')

#-----------FILTER------------------
HAMPEL_WINDOW = 5
HAMPEL_N_SIGMAS = 2
ROLLING_WINDOW = 30

'''
MODEL = 'l2'
PEN = 1000
MINL = 360
JUMP = 1200

'''
#------------Sequence Config------------------
WINDOW = 360  

# inference.py
import time
import datetime
from collections import deque
from pathlib import Path
from typing import cast

import numpy as np
import joblib
from keras import Model, models
from xgboost import XGBRegressor

import firebase_admin
from firebase_admin import credentials, db

from src import config as cf


# ==================== SETTINGS ====================

WINDOW_SIZE = 360                 # 1-hour window (360 samples)
POLL_SECONDS = 10                  # how often to poll Firebase (in seconds)
PREDICTION_INTERVAL_MIN = 10      # only predict every 10 minutes

MODELS = Path(cf.MODELS_DIR)
EMBEDDING_LAYER_NAME = "encoder"  # from your get_embeddings() code

# Firebase → internal feature-name mapping
FIREBASE_TO_INTERNAL = {
    "temperature": "temp",
    "humidity":   "humidity",
    "TVOC":       "tvoc",
    "eCO2":       "eCO2",
    "PM1":        "pm1",
    "PM2_5":      "pm2.5",
    "PM10":       "pm10",
}


# ==================== FIREBASE SETUP ====================

def init_firebase():
    cred = credentials.Certificate(cf.FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": cf.FIREBASE_DB_URL})

    sensor_ref = db.reference("sensorData")
    pred_ref   = db.reference("prediction")  # change name if you want a different node
    return sensor_ref, pred_ref


# ==================== LOAD MODELS ====================

def load_models():
    # scaler + PCA
    scaler = joblib.load(MODELS / "scaler.joblib")
    pca    = joblib.load(MODELS / "pca.joblib")

    # full autoencoder → encoder (layer "encoder")
    autoencoder = cast(Model, models.load_model(MODELS / "lstm_autoencoder.keras"))
    enc_layer   = autoencoder.get_layer(EMBEDDING_LAYER_NAME)
    encoder     = Model(autoencoder.input, enc_layer.output)

    # XGBoost regressors
    m_clean = XGBRegressor()
    m_clean.load_model(MODELS / "xgb_clean.json")

    m_medium = XGBRegressor()
    m_medium.load_model(MODELS / "xgb_medium.json")

    m_high = XGBRegressor()
    m_high.load_model(MODELS / "xgb_high.json")

    return scaler, pca, encoder, m_clean, m_medium, m_high


# ==================== FETCH SAMPLE FROM FIREBASE ====================

def fetch_sample(sensor_ref):
    """
    Read the latest snapshot from /sensorData.
    Returns (sample_vector, last_log_string) or (None, None) if not ready.
    """
    data = sensor_ref.get()
    if not data:
        return None, None

    last_log = data.get("lastLog")
    if last_log is None:
        # We rely on lastLog to detect new readings
        return None, None

    # Map Firebase keys → internal names
    values = {}
    for fb_key, internal in FIREBASE_TO_INTERNAL.items():
        if fb_key not in data:
            return None, None
        values[internal] = float(data[fb_key])

    # Order as in cf.SENSOR_COLUMNS (the training order)
    arr = [values[name] for name in cf.SENSOR_COLUMNS]
    sample = np.array(arr, dtype=float)  # shape (7,)

    return sample, last_log


# ==================== SCALING ====================

def scale_sequence(seq_3d, scaler):
    """
    seq_3d: (1, T, F)
    Returns scaled sequence with the same shape.
    """
    _, T, F = seq_3d.shape
    seq_2d = seq_3d.reshape(T, F)
    seq_scaled_2d = scaler.transform(seq_2d)
    return seq_scaled_2d.reshape(1, T, F)


# ==================== PREDICTION CORE ====================

def predict_from_buffer(buffer,
                        scaler,
                        pca,
                        encoder,
                        m_clean,
                        m_medium,
                        m_high):

    # Build sequence (T, F) → (1, T, F)
    seq = np.stack(buffer, axis=0)           # (T, F)
    seq = seq.reshape(1, WINDOW_SIZE, -1)    # (1, T, F)

    # 1) scale
    seq_scaled = scale_sequence(seq, scaler)

    # 2) LSTM encoder -> embedding
    emb = encoder.predict(seq_scaled, verbose=0)  # (1, emb_dim)

    # 3) PCA -> (1, n_pca)
    z = pca.transform(emb)

    # 4) XGBoost regressors
    mu_clean  = float(m_clean.predict(z)[0])
    mu_medium = float(m_medium.predict(z)[0])
    mu_high   = float(m_high.predict(z)[0])

    # Clip to [0, 1] for interpretability
    mu_clean  = max(0.0, min(1.0, mu_clean))
    mu_medium = max(0.0, min(1.0, mu_medium))
    mu_high   = max(0.0, min(1.0, mu_high))

    mus = np.array([mu_clean, mu_medium, mu_high], dtype=float)
    labels = ["clean", "medium", "high"]
    label = labels[int(mus.argmax())]

    return mu_clean, mu_medium, mu_high, label


# ==================== REALTIME LOOP ====================

def main():
    sensor_ref, pred_ref = init_firebase()
    scaler, pca, encoder, m_clean, m_medium, m_high = load_models()

    buffer = deque(maxlen=WINDOW_SIZE)

    last_log_seen       = None   # last lastLog we used to fill buffer
    last_log_predicted  = None   # last lastLog for which we predicted
    last_prediction_ts  = None   # real time when we last predicted

    print("Realtime inference started...")

    while True:
        sample, last_log = fetch_sample(sensor_ref)

        if sample is not None and last_log is not None:

            # --- Update sliding window  ---
            if last_log != last_log_seen:
                buffer.append(sample)
                last_log_seen = last_log
                print(f"Buffer: {len(buffer)}/{WINDOW_SIZE} | lastLog = {last_log}")


            if len(buffer) == WINDOW_SIZE and last_log != last_log_predicted:
                now = datetime.datetime.now()

                if (last_prediction_ts is None or
                    (now - last_prediction_ts).total_seconds() >= PREDICTION_INTERVAL_MIN * 60):

                    mu_c, mu_m, mu_h, label = predict_from_buffer(
                        buffer, scaler, pca, encoder,
                        m_clean, m_medium, m_high
                    )

                    payload = {
                        "clean":    mu_c,
                        "medium":   mu_m,
                        "high":     mu_h,
                        "label":    label,
                        "lastLog":  last_log,  
                        "generated_at": now.isoformat(timespec="seconds"),
                    }

                    pred_ref.set(payload)
                    print("Prediction sent:", payload)

                    last_prediction_ts = now
                    last_log_predicted = last_log

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

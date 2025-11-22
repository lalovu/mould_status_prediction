# inference.py
import time
from collections import deque
from pathlib import Path

import numpy as np
import joblib
from typing import cast
from keras import Model, models
from xgboost import XGBRegressor

import firebase_admin
from firebase_admin import credentials, db

from src import config as cf


# ==== settings ====
WINDOW_SIZE = 360        # 1-hour window (360 samples)
POLL_SECONDS = 2           # how often to read from Firebase
MODELS = Path(cf.MODELS_DIR)
EMBEDDING_LAYER_NAME = "encoder"   # from your get_embeddings() code

# Firebase → internal feature-name mapping
FIREBASE_TO_INTERNAL = {
    "temperature": "temp",
    "humidity": "humidity",
    "TVOC": "tvoc",
    "eCO2": "eCO2",
    "PM1": "pm1",
    "PM2_5": "pm2.5",
    "PM10": "pm10",
}


# ==== firebase setup ====
def init_firebase():
    cred = credentials.Certificate(cf.FIREBASE_CRED_PATH)
    firebase_admin.initialize_app(cred, {"databaseURL": cf.FIREBASE_DB_URL})

    sensor_ref = db.reference("sensorData")
    pred_ref = db.reference("prediction")   # change node name if you like
    return sensor_ref, pred_ref


# ==== load models ====
def load_models():
    scaler = joblib.load(MODELS / "scaler.joblib")
    pca = joblib.load(MODELS / "pca.joblib")

    # Keras Model
    autoencoder = cast(Model, models.load_model(MODELS / "lstm_autoencoder.keras"))

    enc_layer = autoencoder.get_layer(EMBEDDING_LAYER_NAME)
    encoder = Model(autoencoder.input, enc_layer.output)

    m_clean = XGBRegressor()
    m_clean.load_model(MODELS / "xgb_clean.json")

    m_medium = XGBRegressor()
    m_medium.load_model(MODELS / "xgb_medium.json")

    m_high = XGBRegressor()
    m_high.load_model(MODELS / "xgb_high.json")

    return scaler, pca, encoder, m_clean, m_medium, m_high

# ==== read one sample from firebase ====
def fetch_sample(sensor_ref):
    data = sensor_ref.get()
    if not data:
        return None, None

    # read lastLog to detect new samples
    last_log = data.get("lastLog")
    if last_log is None:
        # fallback: no timestamp, we can't safely detect new data
        return None, None

    # map firebase keys -> internal names
    values = {}
    for fb_key, internal in FIREBASE_TO_INTERNAL.items():
        if fb_key not in data:
            return None, None
        values[internal] = float(data[fb_key])

    # order as in cf.SENSOR_COLUMNS
    arr = [values[name] for name in cf.SENSOR_COLUMNS]
    sample = np.array(arr, dtype=float)  # shape (7,)

    return sample, last_log



# ==== scaling helper ====
def scale_sequence(seq_3d, scaler):
    # seq_3d: (1, T, F)
    _, T, F = seq_3d.shape
    seq_2d = seq_3d.reshape(T, F)
    seq_scaled_2d = scaler.transform(seq_2d)
    return seq_scaled_2d.reshape(1, T, F)


# ==== main prediction from a full buffer ====
def predict_from_buffer(buffer, scaler, pca, encoder,
                        m_clean, m_medium, m_high):
    # buffer: deque of WINDOW_SIZE vectors, each (F,)
    seq = np.stack(buffer, axis=0)             # (T, F)
    seq = seq.reshape(1, WINDOW_SIZE, -1)      # (1, T, F)

    # 1) scale
    seq_scaled = scale_sequence(seq, scaler)

    # 2) LSTM encoder -> embedding
    emb = encoder.predict(seq_scaled, verbose=0)   # (1, emb_dim)

    # 3) PCA
    z = pca.transform(emb)                        # (1, n_pca)

    # 4) XGBoost regressors
    mu_clean = float(m_clean.predict(z)[0])
    mu_medium = float(m_medium.predict(z)[0])
    mu_high = float(m_high.predict(z)[0])

    # clip to [0, 1] just to be safe
    mu_clean = max(0.0, min(1.0, mu_clean))
    mu_medium = max(0.0, min(1.0, mu_medium))
    mu_high = max(0.0, min(1.0, mu_high))

    mus = np.array([mu_clean, mu_medium, mu_high])
    labels = ["clean", "medium", "high"]
    label = labels[int(mus.argmax())]

    return mu_clean, mu_medium, mu_high, label


# ==== realtime loop ====
def main():
    sensor_ref, pred_ref = init_firebase()
    scaler, pca, encoder, m_clean, m_medium, m_high = load_models()

    buffer = deque(maxlen=WINDOW_SIZE)
    last_log_seen = None

    print("Realtime inference started...")

    while True:
        sample, last_log = fetch_sample(sensor_ref)

        if sample is not None and last_log is not None:
            # append only when lastLog changes → new reading arrived
            if last_log != last_log_seen:
                buffer.append(sample)
                last_log_seen = last_log
                print(f"Buffer: {len(buffer)}/{WINDOW_SIZE}")

            if len(buffer) == WINDOW_SIZE:
                mu_c, mu_m, mu_h, label = predict_from_buffer(
                    buffer, scaler, pca, encoder,
                    m_clean, m_medium, m_high
                )

                payload = {
                    "clean": mu_c,
                    "medium": mu_m,
                    "high":  mu_h,
                    "label": label
                }
                pred_ref.set(payload)
                print("Prediction sent:", payload)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

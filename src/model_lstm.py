# model_lstm.py
import numpy as np
from pathlib import Path

from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from joblib import dump


def load_data(splits_path: Path):
    """Load train/val/test window arrays from disk."""
    x_tr = np.load(splits_path / "train_windows.npy")
    x_va = np.load(splits_path / "val_windows.npy")
    x_te = np.load(splits_path / "test_windows.npy")
    return x_tr, x_va, x_te


def scale_data(x_tr, x_va, x_te):
    """Standardize features using train statistics."""
    n_tr, t, f = x_tr.shape
    scaler = StandardScaler()

    x_tr = scaler.fit_transform(x_tr.reshape(-1, f)).reshape(n_tr, t, f)
    x_va = scaler.transform(x_va.reshape(-1, f)).reshape(x_va.shape[0], t, f)
    x_te = scaler.transform(x_te.reshape(-1, f)).reshape(x_te.shape[0], t, f)

    return x_tr, x_va, x_te, scaler


def build_autoencoder(input_shape, emb_dim: int):
    """Simple LSTM autoencoder."""
    t, f = input_shape
    inp = layers.Input(shape=input_shape)
    z = layers.LSTM(emb_dim, name="encoder")(inp)
    d = layers.RepeatVector(t)(z)
    d = layers.LSTM(emb_dim, return_sequences=True)(d)
    out = layers.TimeDistributed(layers.Dense(f))(d)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_metrics(x_true, x_pred, name: str):
    diff = x_true - x_pred
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    a = x_true.reshape(-1)
    b = x_pred.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(a, b)[0, 1]

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")


def get_embeddings(model, x):
    """Return encoder output for each sequence."""
    encoder = models.Model(model.input, model.get_layer("encoder").output)
    return encoder.predict(x, verbose='0')


def run_lstm(
    splits_dir: str = "splits",
    emb_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 8,
):
    """Train LSTM autoencoder and save scaler, model, and embeddings."""
    path = Path(splits_dir)
    path.mkdir(exist_ok=True)

    x_tr, x_va, x_te = load_data(path)
    x_tr, x_va, x_te, scaler = scale_data(x_tr, x_va, x_te)
    dump(scaler, path / "scaler.joblib")

    model = build_autoencoder(x_tr.shape[1:], emb_dim)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
        verbose= 1,
    )

    model.fit(
        x_tr,
        x_tr,
        validation_data=(x_va, x_va),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose= '1',
    )

    tr_rec = model.predict(x_tr, verbose='0')
    va_rec = model.predict(x_va, verbose='0')
    te_rec = model.predict(x_te, verbose='0')

    reconstruction_metrics(x_tr, tr_rec, "Train")
    reconstruction_metrics(x_va, va_rec, "Val")
    reconstruction_metrics(x_te, te_rec, "Test")

    tr_emb = get_embeddings(model, x_tr)
    va_emb = get_embeddings(model, x_va)
    te_emb = get_embeddings(model, x_te)

    model.save(path / "lstm_autoencoder.keras")
    np.save(path / "train_embeddings.npy", tr_emb)
    np.save(path / "val_embeddings.npy", va_emb)
    np.save(path / "test_embeddings.npy", te_emb)

    print("Saved to", path)
    print("  scaler.joblib")
    print("  lstm_autoencoder.keras")
    print("  train_embeddings.npy, val_embeddings.npy, test_embeddings.npy")


if __name__ == "__main__":
    run_lstm()

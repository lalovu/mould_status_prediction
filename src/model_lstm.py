# lstm.py
import numpy as np
from pathlib import Path
from keras import layers, models
from sklearn.preprocessing import StandardScaler
from joblib import dump


def load_data(splits_path: Path):
    """Load train/val/test windows from splits folder."""
    X_train = np.load(splits_path / "train_windows.npy")
    X_val   = np.load(splits_path / "val_windows.npy")
    X_test  = np.load(splits_path / "test_windows.npy")
    return X_train, X_val, X_test


def scale_data(X_train, X_val, X_test):
    """Standardize features using training statistics and return scaler."""
    n_train, t, f = X_train.shape
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.reshape(-1, f)).reshape(n_train, t, f)
    X_val   = scaler.transform(X_val.reshape(-1, f)).reshape(X_val.shape[0], t, f)
    X_test  = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape[0], t, f)

    return X_train, X_val, X_test, scaler


def build_autoencoder(input_shape, emb_dim: int):
    """LSTM autoencoder: sequence -> embedding -> reconstructed sequence."""
    t, f = input_shape
    x = layers.Input(shape=input_shape, name="sequence_input")
    z = layers.LSTM(emb_dim, name="encoder_lstm")(x)
    d = layers.RepeatVector(t)(z)
    d = layers.LSTM(emb_dim, return_sequences=True)(d)
    y = layers.TimeDistributed(layers.Dense(f), name="reconstruction")(d)
    model = models.Model(x, y, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def metrics(y_true, y_pred, name: str):
    """Print MAE, RMSE, Pearson correlation."""
    diff = y_true - y_pred
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))

    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        corr = np.nan
    else:
        corr = np.corrcoef(a, b)[0, 1]

    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, Corr={corr:.4f}")


def get_embeddings(model, X):
    """Extract encoder LSTM output as temporal embeddings."""
    encoder = models.Model(
        inputs=model.input,
        outputs=model.get_layer("encoder_lstm").output,
        name="encoder_model",
    )
    return encoder.predict(X, verbose='0')


def run_lstm(
    splits_dir: str = "splits",
    emb_dim: int = 32,
    epochs: int = 20,
    batch_size: int = 32,
):
    """
    Train LSTM autoencoder on sequences, evaluate reconstruction,
    save scaler, model, and embeddings.
    """
    splits_path = Path(splits_dir)
    splits_path.mkdir(exist_ok=True)

    # 1) data
    X_train, X_val, X_test = load_data(splits_path)
    X_train, X_val, X_test, scaler = scale_data(X_train, X_val, X_test)

    # save scaler for inference
    dump(scaler, splits_path / "scaler.joblib")

    # 2) model
    model = build_autoencoder(X_train.shape[1:], emb_dim)
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        verbose='1',  #
    )

    # 3) reconstruction + metrics
    train_rec = model.predict(X_train, verbose='0')
    val_rec   = model.predict(X_val,   verbose='0')
    test_rec  = model.predict(X_test,  verbose='0')

    metrics(X_train, train_rec, "Train")
    metrics(X_val,   val_rec,   "Val")
    metrics(X_test,  test_rec,  "Test")

    # 4) embeddings
    train_emb = get_embeddings(model, X_train)
    val_emb   = get_embeddings(model, X_val)
    test_emb  = get_embeddings(model, X_test)

    # 5) save model + embeddings
    model.save(splits_path / "lstm_autoencoder.keras")
    np.save(splits_path / "train_embeddings.npy", train_emb)
    np.save(splits_path / "val_embeddings.npy",   val_emb)
    np.save(splits_path / "test_embeddings.npy",  test_emb)

    print("Saved:")
    print(" - scaler.joblib")
    print(" - lstm_autoencoder.keras")
    print(" - train/val/test_embeddings.npy in", splits_path)


if __name__ == "__main__":
    run_lstm()

# lstm.py
import numpy as np
from keras import layers, models
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from pathlib import Path

SPLITS = "splits"
OUT_DIM = 32      
PCA_DIM = 10      

def load_data():    
    X_train = np.load(f"{SPLITS}/X_train.npy")
    X_val = np.load(f"{SPLITS}/X_val.npy")
    X_test = np.load(f"{SPLITS}/X_test.npy")
    return X_train, X_val, X_test

def normalize_inputs(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    return X_train, X_val, X_test

def build_lstm(input_shape, embedding_dim=32, output_dim=7):
    x_in = layers.Input(shape=input_shape)
    z    = layers.LSTM(embedding_dim, return_sequences=False, name="lstm")(x_in)
    y    = layers.Dense(output_dim, activation="linear")(z)
    model = models.Model(inputs=x_in, outputs=y)
    model.compile(optimizer="adam", loss="mse")
    return model

def extract_embeddings(model, X):
    return model.predict(X, verbose=0)

def main():
    Path(SPLITS).mkdir(exist_ok=True)
    X_train, X_val, X_test = load_data()
    X_train, X_val, X_test = normalize_inputs(X_train, X_val, X_test)

    model = build_lstm(X_train.shape[1:])
    model.fit(X_train, X_train, epochs=20, batch_size=32, validation_data=(X_val, X_val), verbose='1')

    train_emb = extract_embeddings(model, X_train)
    val_emb   = extract_embeddings(model, X_val)
    test_emb  = extract_embeddings(model, X_test)

    # normalize embeddings for FCM
    train_emb = normalize(train_emb, axis=1)
    val_emb   = normalize(val_emb, axis=1)
    test_emb  = normalize(test_emb, axis=1)

    # optional PCA reduction
    pca = PCA(n_components=PCA_DIM, random_state=0)
    train_emb = pca.fit_transform(train_emb)
    val_emb   = pca.transform(val_emb)
    test_emb  = pca.transform(test_emb)

    np.save(f"{SPLITS}/train_embeddings.npy", train_emb)
    np.save(f"{SPLITS}/val_embeddings.npy", val_emb)
    np.save(f"{SPLITS}/test_embeddings.npy", test_emb)

    print("Embeddings saved ✓")
    print(f"Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}")

if __name__ == "__main__":
    main()

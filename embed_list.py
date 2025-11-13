# embed_list.py 
from pathlib import Path
import numpy as np
from keras import models                        
from lstm import build_lstm                       
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

SPLITS_DIR = "splits"
OUT_DIM = 32
PCA_DIM = 10
BATCH_SIZE = 32
EPOCHS = 20

def load_split(name):
    x = np.load(f"{SPLITS_DIR}/{name}_windows.npy").astype(np.float32)
    assert x.ndim == 3, f"{name} shape mismatch: {x.shape}"
    return x

def normalize_inputs(train, val, test):
    sc = StandardScaler()
    train = sc.fit_transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    val   = sc.transform(val.reshape(-1, val.shape[-1])).reshape(val.shape)
    test  = sc.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    return train, val, test

def main():
    Path(SPLITS_DIR).mkdir(exist_ok=True)

    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    train, val, test = normalize_inputs(train, val, test)

    # build & train to predict last timestep (target shape = (batch, 7))
    model = build_lstm(input_shape=train.shape[1:], embedding_dim=OUT_DIM, output_dim=train.shape[-1])

    y_train = train[:, -1, :]   # last timestep
    y_val   = val[:, -1, :]

    model.fit(
        train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(val, y_val),
        verbose= '1'  # or "1" if your linter complains
    )

    # ----- EMBEDDING EXTRACTION -----
    # model.layers = [InputLayer, LSTM, Dense]; take index 1
   
    embedding_model = models.Model(
        inputs=model.inputs,
        outputs=model.get_layer("lstm").output
    )

    train_emb = embedding_model.predict(train, verbose='0')
    val_emb   = embedding_model.predict(val,   verbose='0')
    test_emb  = embedding_model.predict(test,  verbose='0')

    # normalize + (optional) PCA for FCM stability
    train_emb = normalize(train_emb, axis=1)
    val_emb   = normalize(val_emb, axis=1)
    test_emb  = normalize(test_emb, axis=1)

    pca = PCA(n_components=PCA_DIM, random_state=0)
    train_emb = pca.fit_transform(train_emb)
    val_emb   = pca.transform(val_emb)
    test_emb  = pca.transform(test_emb)

    np.save(f"{SPLITS_DIR}/train_embeddings.npy", train_emb)
    np.save(f"{SPLITS_DIR}/val_embeddings.npy",   val_emb)
    np.save(f"{SPLITS_DIR}/test_embeddings.npy",  test_emb)

    print("Done:", train_emb.shape, val_emb.shape, test_emb.shape)

if __name__ == "__main__":
    main()

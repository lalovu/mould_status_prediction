# run_models.py
from src.model_lstm import run_lstm
from src.pca_embed import run_pca


def run_models():
    splits_dir = "splits"

    # Phase 1: LSTM autoencoder (temporal embeddings)
    run_lstm(
        splits_dir=splits_dir,
        emb_dim=32,
        epochs=20,
        batch_size=32,
    )

    # Phase 2: PCA (+ KMO, Bartlett) on embeddings
    run_pca(
        splits_dir=splits_dir,
        pca_dim=10,
    )

    # Later: add XGBoost, FCM, etc. here using outputs from splits_dir.


if __name__ == "__main__":
    run_models()

# run_models.py

from src.model_lstm import run_lstm
from src.pca_embed import run_pca


def run_models():
    splits_dir = "splits"

    # 1) LSTM autoencoder → temporal embeddings
    run_lstm(
        splits_dir=splits_dir,
        emb_dim=32,     
        epochs=100,     
        batch_size=32,
        
    )

    # 2) PCA on embeddings (KMO + Bartlett + PCA)
    run_pca(
        splits_dir=splits_dir,
        pca_dim=10,
    )


if __name__ == "__main__":
    run_models()

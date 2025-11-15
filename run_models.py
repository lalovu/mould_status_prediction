# run_models.py

# If your files are inside src/, keep these:
from src.model_lstm import run_lstm
from src.pca_embed import run_pca


def run_models():
    splits_dir = "splits"

    # 1) LSTM autoencoder → temporal embeddings
    run_lstm(
        splits_dir=splits_dir,
        emb_dim=32,     # or 64 if you want a larger embedding
        epochs=100,     # EarlyStopping will stop earlier if you added it
        batch_size=32,
        # patience uses the default from model_lstm.run_lstm if you kept that arg
    )

    # 2) PCA on embeddings (KMO + Bartlett + PCA)
    run_pca(
        splits_dir=splits_dir,
        pca_dim=10,
    )


if __name__ == "__main__":
    run_models()

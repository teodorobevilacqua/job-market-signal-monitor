# src/embeddings.py
from pathlib import Path
import numpy as np
import pandas as pd

def load_embeddings(emb_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    X_path = emb_dir / "job_embeddings.npy"
    meta_path = emb_dir / "job_embeddings_meta.parquet"

    X = np.load(X_path)
    meta = pd.read_parquet(meta_path)

    if len(meta) != X.shape[0]:
        raise ValueError(f"Meta rows ({len(meta)}) != embeddings rows ({X.shape[0]})")

    return X, meta
# tests/test_smoke.py
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR, EMB_DIR, DEFAULT_K


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Missing artifact: {path}")


def test_processed_dataset_loads():
    """
    Smoke test: processed dataset can be loaded (at least one parquet present).
    """
    _skip_if_missing(PROCESSED_DIR)

    parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
    if not parquet_files:
        pytest.skip(f"No parquet files found in {PROCESSED_DIR}")

    df = pd.read_parquet(parquet_files[0])
    assert df.shape[0] > 0
    assert df.shape[1] > 0


def test_embeddings_match_metadata_rows():
    """
    Smoke test: embeddings array rows match metadata rows.
    """
    X_path = EMB_DIR / "job_embeddings.npy"
    meta_path = EMB_DIR / "job_embeddings_meta.parquet"

    _skip_if_missing(X_path)
    _skip_if_missing(meta_path)

    X = np.load(X_path)
    meta = pd.read_parquet(meta_path)

    assert X.ndim == 2
    assert len(meta) == X.shape[0]


def test_kmeans_produces_k_clusters():
    """
    Smoke test: KMeans produces exactly K unique labels on a small sample.
    """
    from src.embeddings import load_embeddings
    from src.clustering import fit_kmeans

    _skip_if_missing(EMB_DIR / "job_embeddings.npy")
    _skip_if_missing(EMB_DIR / "job_embeddings_meta.parquet")

    X, _ = load_embeddings(EMB_DIR)

    # Keep it fast: sample max 5000 rows
    n = min(5000, X.shape[0])
    Xs = X[:n]

    labels = fit_kmeans(Xs, k=DEFAULT_K, n_init=3, random_state=42)

    assert len(labels) == n
    assert len(set(labels)) == DEFAULT_K
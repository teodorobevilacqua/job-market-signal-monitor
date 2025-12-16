# src/clustering.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def fit_kmeans(X: np.ndarray, k: int, n_init: int = 10, random_state: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return labels

def auto_cluster_labels(meta: pd.DataFrame, label_col: str = "cluster", title_col: str = "title") -> dict:
    top_titles = (
        meta.groupby(label_col)[title_col]
        .apply(lambda s: s.value_counts().head(5).index.tolist())
    )
    return {c: " | ".join(titles[:2]) for c, titles in top_titles.items()}

def apply_cluster_labels(
    meta: pd.DataFrame,
    labels: np.ndarray,
    manual_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    out = meta.copy()
    out["cluster"] = labels

    auto_map = auto_cluster_labels(out, label_col="cluster", title_col="title")
    out["cluster_label"] = out["cluster"].map(auto_map)

    if manual_map is None:
        out["cluster_label_final"] = out["cluster_label"]
    else:
        out["cluster_label_manual"] = out["cluster"].map(manual_map)
        out["cluster_label_final"] = out["cluster_label_manual"].fillna(out["cluster_label"])

    return out

def save_clustered_meta(meta: pd.DataFrame, out_path: Path) -> None:
    meta.to_parquet(out_path, index=False)
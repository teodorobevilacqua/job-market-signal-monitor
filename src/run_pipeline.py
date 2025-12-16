# src/run_pipeline.py
from pathlib import Path
import pandas as pd

from src.config import EMB_DIR, CLUST_DIR, DOCS_DIR, IMG_DIR, DEFAULT_K, RANDOM_STATE
from src.io_utils import ensure_dir
from src.embeddings import load_embeddings
from src.clustering import fit_kmeans, apply_cluster_labels, save_clustered_meta
from src.signals import add_period_from_date, compute_delta_share, top_rising_declining
from src.plots import save_delta_share_bar

def main() -> None:
    # 1) Load embeddings + meta
    X, meta = load_embeddings(EMB_DIR)

    # 2) Fit clustering
    labels = fit_kmeans(X, k=DEFAULT_K, n_init=5, random_state=RANDOM_STATE)

    # 3) Apply labels (manual map optional)
    manual_map = {
        # fill with your real IDs if you want:
        # 12: "Engineering Roles (Electrical & Mechanical)",
        # 7: "Clinical Support & Medical Assistant Roles",
    }
    meta = apply_cluster_labels(meta, labels, manual_map=manual_map if manual_map else None)

    # 4) Add period + compute signals
    meta = add_period_from_date(meta, date_col="date")
    pivot = compute_delta_share(meta, label_col="cluster_label_final")
    signal_summary = top_rising_declining(pivot, n=5)

    # 5) Save artifacts
    ensure_dir(CLUST_DIR)
    ensure_dir(DOCS_DIR)
    ensure_dir(IMG_DIR)

    clustered_path = CLUST_DIR / "job_postings_with_clusters.parquet"
    save_clustered_meta(meta, clustered_path)

    csv_path = DOCS_DIR / "top_rising_declining_clusters.csv"
    signal_summary_rounded = signal_summary.copy()
    for c in ["early_period", "late_period", "delta_share"]:
        if c in signal_summary_rounded.columns:
            signal_summary_rounded[c] = signal_summary_rounded[c].round(3)
    signal_summary_rounded.to_csv(csv_path, index=False)

    fig_path = IMG_DIR / "delta_share_top_movers.png"
    # set index name for plotting nicely
    pivot_for_plot = pivot.copy()
    pivot_for_plot.index.name = "Job Family"
    save_delta_share_bar(pivot_for_plot, fig_path, top_n=10, index_name="Job Family")

    print("Saved:")
    print(" -", clustered_path)
    print(" -", csv_path)
    print(" -", fig_path)

if __name__ == "__main__":
    main()
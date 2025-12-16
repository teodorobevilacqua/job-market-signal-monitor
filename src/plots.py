# src/plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_delta_share_bar(
    pivot: pd.DataFrame,
    out_path: Path,
    top_n: int = 10,
    index_name: str = "Job Family",
) -> Path:
    # pivot index = cluster_label_final; must contain 'delta_share'
    df = pivot[["delta_share"]].reset_index().rename(columns={pivot.index.name or "index": index_name})

    top_rise = df.nlargest(top_n, "delta_share")
    top_fall = df.nsmallest(top_n, "delta_share")
    plot_df = pd.concat([top_rise, top_fall], axis=0)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[index_name], plot_df["delta_share"])
    plt.axvline(0)
    plt.title(f"Job Family Share Change (Late − Early): Top {top_n} Rising & Declining")
    plt.xlabel("Δ share")
    plt.ylabel("")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path
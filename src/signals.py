# src/signals.py
import pandas as pd

def add_period_from_date(meta: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = meta.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])

    split_date = out[date_col].min() + (out[date_col].max() - out[date_col].min()) / 2
    out["period"] = out[date_col].apply(lambda d: "early_period" if d <= split_date else "late_period")
    return out

def compute_delta_share(meta: pd.DataFrame, label_col: str = "cluster_label_final") -> pd.DataFrame:
    if "period" not in meta.columns:
        raise ValueError("meta must contain a 'period' column. Use add_period_from_date() first.")

    cluster_period = (
        meta.groupby("period")[label_col]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
    )

    pivot = (
        cluster_period
        .pivot(index=label_col, columns="period", values="share")
        .fillna(0)
    )

    pivot["delta_share"] = pivot["late_period"] - pivot["early_period"]
    pivot = pivot.sort_values("delta_share", ascending=False)

    return pivot

def top_rising_declining(pivot: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    top_rise = pivot.head(n).copy()
    top_fall = pivot.tail(n).copy()

    top_rise["signal"] = "Rising"
    top_fall["signal"] = "Declining"

    out = pd.concat([top_rise, top_fall], axis=0).reset_index()
    return out
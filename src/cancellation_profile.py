import os
import pandas as pd
import numpy as np


def cancellation_profile_by_cluster(
    df_original: pd.DataFrame,
    labels,
    method: str,
    representation_id: str,):
   

    labels = np.asarray(labels)

    if len(df_original) != len(labels):
        raise ValueError(
            f"Length mismatch: df_original has {len(df_original)} rows, "
            f"but labels has {len(labels)} rows."
        )

    if "is_canceled" not in df_original.columns:
        raise ValueError("df_original must contain the column 'is_canceled'.")

    profile_df = df_original.copy()
    profile_df["cluster"] = labels

    global_cancellation_rate = profile_df["is_canceled"].mean()

    table = (
        profile_df
        .groupby("cluster")
        .agg(
            n=("is_canceled", "size"),
            canceled_count=("is_canceled", "sum"),
            cancellation_rate=("is_canceled", "mean"),
        )
        .reset_index()
    )

    table["not_canceled_count"] = table["n"] - table["canceled_count"]
    table["pct_total"] = table["n"] / len(profile_df)
    table["global_cancellation_rate"] = global_cancellation_rate
    table["difference_from_global"] = (
        table["cancellation_rate"] - global_cancellation_rate
    )

    table["method"] = method
    table["representation_id"] = representation_id

    table = table[
        [
            "method",
            "representation_id",
            "cluster",
            "n",
            "pct_total",
            "canceled_count",
            "not_canceled_count",
            "cancellation_rate",
            "global_cancellation_rate",
            "difference_from_global",
        ]
    ]

    return table.sort_values("cluster").reset_index(drop=True)


def save_cancellation_profile(
    table: pd.DataFrame,
    filename: str,
    output_dir: str = "tables/clusterProfiles",):
    

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    table.to_csv(output_path, index=False)


def compute_and_save_cancellation_profile(
    df_original: pd.DataFrame,
    labels,
    method: str,
    representation_id: str,
    filename: str,
    output_dir: str = "tables/clusterProfiles",):
    

    table = cancellation_profile_by_cluster(
        df_original=df_original,
        labels=labels,
        method=method,
        representation_id=representation_id,
    )

    save_cancellation_profile(
        table=table,
        filename=filename,
        output_dir=output_dir,
    )

    return table
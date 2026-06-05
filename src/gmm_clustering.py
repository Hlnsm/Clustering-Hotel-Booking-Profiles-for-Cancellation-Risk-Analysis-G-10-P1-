import time
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


def safe_internal_metrics(X, labels):
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))

    if n_clusters < 2 or n_clusters >= len(labels):
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }


def compute_ari_by_k_and_covariance(all_runs):
    rows = []

    for (covariance_type, K), group in all_runs.groupby(["covariance_type", "K"]):
        labels_list = list(group["labels"])

        if len(labels_list) < 2:
            rows.append({
                "covariance_type": covariance_type,
                "K": K,
                "ari_mean": np.nan,
                "ari_std": np.nan,
            })
            continue

        ari_values = [
            adjusted_rand_score(a, b)
            for a, b in combinations(labels_list, 2)
        ]

        rows.append({
            "covariance_type": covariance_type,
            "K": K,
            "ari_mean": float(np.mean(ari_values)),
            "ari_std": float(np.std(ari_values, ddof=1)) if len(ari_values) > 1 else 0.0,
        })

    return pd.DataFrame(rows)


def explore_k_values_gmm(
    X,
    K_values,
    M=5,
    covariance_types=("diag",),
    max_iter=300,
    tol=1e-3,
    reg_covar=1e-6,
    base_seed=3000,
):
    

    X = np.asarray(X, dtype=np.float64)
    rows = []

    for covariance_type in covariance_types:
        for K in K_values:
            for run in range(1, M + 1):
                seed = base_seed + 100 * K + run

                start = time.perf_counter()

                model = GaussianMixture(
                    n_components=K,
                    covariance_type=covariance_type,
                    n_init=1,
                    max_iter=max_iter,
                    tol=tol,
                    reg_covar=reg_covar,
                    random_state=seed,
                )

                model.fit(X)

                runtime_seconds = time.perf_counter() - start

                labels = model.predict(X)
                membership = model.predict_proba(X)
                max_membership = np.max(membership, axis=1)

                metrics = safe_internal_metrics(X, labels)

                cluster_sizes = [
                    int(np.sum(labels == cluster_id))
                    for cluster_id in range(K)
                ]

                rows.append({
                    "method": "gmm",
                    "K": K,
                    "run": run,
                    "seed": seed,
                    "covariance_type": covariance_type,
                    "silhouette": metrics["silhouette"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "aic": float(model.aic(X)),
                    "bic": float(model.bic(X)),
                    "n_iterations": int(model.n_iter_),
                    "converged": bool(model.converged_),
                    "avg_max_membership": float(np.mean(max_membership)),
                    "median_max_membership": float(np.median(max_membership)),
                    "pct_ambiguous_0_70": float(np.mean(max_membership < 0.70)),
                    "cluster_sizes": cluster_sizes,

                    # kept in memory only
                    "labels": labels,
                    "membership": membership,
                    "model": model,
                })

    all_runs = pd.DataFrame(rows)

    export_cols = [
    "method",
    "K",
    "run",
    "seed",
    "covariance_type",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "aic",
    "bic",
    "n_iterations",
    "converged",
    "avg_max_membership",
    "pct_ambiguous_0_70",
    "cluster_sizes",]

    all_runs_clean = all_runs[export_cols].copy()

    summary_metrics = [
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "aic",
    "bic",
    "n_iterations",
    "avg_max_membership",
    "pct_ambiguous_0_70",
]
    summary_by_k = (
        all_runs_clean
        .groupby(["covariance_type", "K"])[summary_metrics]
        .agg(["mean", "std"])
        .round(4)
    )

    summary_by_k.columns = [
        f"{metric}_{stat}" for metric, stat in summary_by_k.columns
    ]

    summary_by_k = summary_by_k.reset_index()

    ari_by_k = compute_ari_by_k_and_covariance(all_runs).round(4)

    summary_by_k = summary_by_k.merge(
        ari_by_k,
        on=["covariance_type", "K"],
        how="left",
    )

    best_metric_runs = pd.DataFrame([
        {
            "selection_metric": "silhouette",
            **all_runs_clean.loc[
                all_runs_clean["silhouette"].idxmax()
            ].to_dict(),
        },
        {
            "selection_metric": "calinski_harabasz",
            **all_runs_clean.loc[
                all_runs_clean["calinski_harabasz"].idxmax()
            ].to_dict(),
        },
        {
            "selection_metric": "davies_bouldin",
            **all_runs_clean.loc[
                all_runs_clean["davies_bouldin"].idxmin()
            ].to_dict(),
        },
        {
            "selection_metric": "aic",
            **all_runs_clean.loc[
                all_runs_clean["aic"].idxmin()
            ].to_dict(),
        },
        {
            "selection_metric": "bic",
            **all_runs_clean.loc[
                all_runs_clean["bic"].idxmin()
            ].to_dict(),
        },
    ])

    return all_runs, all_runs_clean, summary_by_k, best_metric_runs
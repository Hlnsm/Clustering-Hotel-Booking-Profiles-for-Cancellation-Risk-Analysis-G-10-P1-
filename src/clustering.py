import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
from src.evaluation import compute_ari_by_k,evaluate_kmeans_run

def kmeans_scratch(X, K=None, max_iter=100, seed=42, initial_centroids=None):
    X = np.asarray(X, dtype=np.float64)

    if initial_centroids is not None:
        centroids = np.asarray(initial_centroids, dtype=np.float64).copy()
        K = centroids.shape[0]
    else:
        np.random.seed(seed)
        initial_indices = np.random.choice(X.shape[0], K, replace=False)
        centroids = X[initial_indices].copy()

    previous_labels = None
    history = []

    for iteration in range(1, max_iter + 1):

        distances = np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        new_centroids = centroids.copy()

        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)

        objective = 0.0

        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                objective += np.sum((cluster_points - new_centroids[k]) ** 2)

        changed_assignments = None
        stop_condition = False

        if previous_labels is not None:
            changed_assignments = int(np.sum(labels != previous_labels))
            stop_condition = changed_assignments == 0

        centroid_shift = float(np.linalg.norm(new_centroids - centroids))

        history.append({
            "iteration": iteration,
            "objective": float(objective),
            "centroid_shift": centroid_shift,
            "changed_assignments": changed_assignments,
            "stop_condition": stop_condition,
        })

        centroids = new_centroids.copy()
        previous_labels = labels.copy()

        if stop_condition:
            break

    return labels, centroids, history


def run_clustering(df):
    X = df.values

    print("NaN:", np.isnan(X).sum())
    print("Inf:", np.isinf(X).sum())

    labels, centroids, history = kmeans_scratch(df, K=5, max_iter=100, seed=42)

    print("Clusters:", np.unique(labels, return_counts=True))
    print("Iterations:", len(history))
    print("Final objective:", history[-1]["objective"])
    print("Converged:", history[-1]["stop_condition"])

    return labels, centroids, history

def explore_k_values_kmeans(X, K_values, M, max_iter=100, base_seed=1000):
    rows = []

    for K in K_values:
        for run in range(1, M + 1):
            seed = base_seed + 100 * K + run

            labels, centroids, history = kmeans_scratch(
                X,
                K=K,
                max_iter=max_iter,
                seed=seed
            )

            metrics = evaluate_kmeans_run(X, labels, centroids, history)

            rows.append({
                "method": "kmeans",
                "K": K,
                "run": run,
                "seed": seed,
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
                "n_iterations": metrics["n_iterations"],
                "converged": metrics["converged"],

                # guardado só em memória, não para CSV
                "labels": labels,
                "centroids": centroids,
                "history": history,
                "cluster_sizes": metrics["cluster_sizes"],
            })

    all_runs = pd.DataFrame(rows)

    export_cols = [
        "method",
        "K",
        "run",
        "seed",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "n_iterations",
        "converged",
    ]

    all_runs_clean = all_runs[export_cols].copy()

    summary_by_k = (
        all_runs_clean
        .groupby("K")[[
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "n_iterations",
        ]]
        .agg(["mean", "std"])
        .round(4)
    )

    
    summary_by_k.columns = [
        f"{metric}_{stat}" for metric, stat in summary_by_k.columns
    ]

    summary_by_k = summary_by_k.reset_index()

    ari_by_k = compute_ari_by_k(all_runs).round(4)

    summary_by_k = summary_by_k.merge(
        ari_by_k,
        on="K",
        how="left"
    )

    best_metric_runs = pd.DataFrame([
        {
            "selection_metric": "silhouette",
            **all_runs_clean.loc[
                all_runs_clean["silhouette"].idxmax()
            ].to_dict()
        },
        {
            "selection_metric": "calinski_harabasz",
            **all_runs_clean.loc[
                all_runs_clean["calinski_harabasz"].idxmax()
            ].to_dict()
        },
        {
            "selection_metric": "davies_bouldin",
            **all_runs_clean.loc[
                all_runs_clean["davies_bouldin"].idxmin()
            ].to_dict()
        },
    ])

    return all_runs_clean, summary_by_k, best_metric_runs

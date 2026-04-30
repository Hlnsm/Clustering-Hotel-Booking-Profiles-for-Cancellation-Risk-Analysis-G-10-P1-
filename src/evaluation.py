from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
import itertools
import numpy as np
import pandas as pd


def evaluate_kmeans_run(X, labels, centroids, history=None):

    if isinstance(X, pd.DataFrame):
        X_eval = X.to_numpy(dtype=float)
    else:
        X_eval = np.asarray(X, dtype=float)

    labels = np.asarray(labels)

    K = len(np.unique(labels))

    # Segurança: estas métricas precisam de pelo menos 2 clusters
    # e menos clusters do que pontos.
    if K < 2 or K >= len(X_eval):
        raise ValueError("As métricas internas precisam de 2 <= K < n_samples.")

    inertia = 0.0
    cluster_sizes = {}

    for k in range(K):
        cluster_points = X_eval[labels == k]
        cluster_sizes[k] = len(cluster_points)

        if len(cluster_points) > 0:
            inertia += np.sum((cluster_points - centroids[k]) ** 2)

    result = {
        "K": K,
        "silhouette": silhouette_score(X_eval, labels),
        "calinski_harabasz": calinski_harabasz_score(X_eval, labels),
        "davies_bouldin": davies_bouldin_score(X_eval, labels),
        "inertia": inertia,
        "n_iterations": len(history) if history is not None else None,
        "cluster_sizes": cluster_sizes,
    }

    if history is not None and len(history) > 0:
        result["final_objective"] = history[-1]["objective"]
        result["final_centroid_shift"] = history[-1]["centroid_shift"]
        result["final_changed_assignments"] = history[-1]["changed_assignments"]
        result["converged"] = history[-1]["stop_condition"]

    return result

def compute_ari_by_k(all_runs):
    
    ari_results = []

    for K in sorted(all_runs["K"].unique()):
        runs_k = all_runs[all_runs["K"] == K]

        labels_list = runs_k["labels"].tolist()

        ari_scores = []

        for l1, l2 in itertools.combinations(labels_list, 2):
            ari_scores.append(adjusted_rand_score(l1, l2))

        if len(ari_scores) > 0:
            ari_mean = np.mean(ari_scores)
            ari_std = np.std(ari_scores)
        else:
            ari_mean = None
            ari_std = None

        ari_results.append({
            "K": K,
            "ari_mean": ari_mean,
            "ari_std": ari_std
        })

    return pd.DataFrame(ari_results)

def centroid_profile(X, centroids, top_n=10, representation_id="unknown"):
    grand_mean = X.mean(axis=0)
    cluster_tables = {}

    for k, centroid in enumerate(centroids):
        diff = centroid - grand_mean.values

        ranking = (
            pd.DataFrame({
                "feature": X.columns,
                "abs_difference": np.abs(diff)
            })
            .sort_values("abs_difference", ascending=False)
            .head(top_n)
        )

        selected_features = ranking["feature"].tolist()
        selected_idx = [X.columns.get_loc(f) for f in selected_features]

        centroid_values = centroid[selected_idx]
        grand_mean_values = grand_mean.values[selected_idx]
        difference_values = centroid_values - grand_mean_values

        table = pd.DataFrame(
            data=[
                centroid_values,
                grand_mean_values,
                difference_values
            ],
            index=[
                "cluster_mean",
                "grand_mean",
                "difference"
            ],
            columns=selected_features
        )

        table.insert(0, "representation_id", representation_id)
        table.insert(1, "cluster", k)

        cluster_tables[k] = table

    return cluster_tables
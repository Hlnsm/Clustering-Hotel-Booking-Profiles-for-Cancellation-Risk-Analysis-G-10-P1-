from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import pandas as pd


def evaluate_kmeans_run(X, labels, centroids, history=None):
    """
    Avalia uma única execução de k-means.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Matriz já preprocessada/standardizada usada no clustering.
    labels : np.ndarray
        Cluster atribuído a cada ponto.
    centroids : np.ndarray
        Centroides finais devolvidos pelo kmeans_scratch.
    history : list[dict], optional
        Histórico devolvido pelo kmeans_scratch.

    Returns
    -------
    dict
        Métricas internas da run.
    """

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
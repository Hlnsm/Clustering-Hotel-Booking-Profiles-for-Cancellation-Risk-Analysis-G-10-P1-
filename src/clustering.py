import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 


def kmeans_scratch(X, K=5, max_iter=100, seed=42):
    X = X.to_numpy(dtype=float)

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
            "changed_assignments": changed_assignments,
            "centroid_shift": centroid_shift,
            "stop_condition": stop_condition
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

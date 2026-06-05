from dataclasses import dataclass
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from src.clustering import kmeans_scratch
from src.evaluation import evaluate_kmeans_run

FloatArray = NDArray[np.float64]
@dataclass(frozen=True)
class APCluster:
    indices: list[int]
    centroid_raw: FloatArray
    centroid_std: FloatArray
    size: int
    scatter_pct: float


def compute_feature_statistics(X: FloatArray,use_unit_ranges: bool = False,) -> tuple[FloatArray, FloatArray, float]:
    X = np.asarray(X, dtype=np.float64)

    mean = X.mean(axis=0)

    if use_unit_ranges:
        ranges = np.ones(X.shape[1], dtype=np.float64)
    else:
        ranges = X.max(axis=0) - X.min(axis=0)
        ranges = np.where(ranges == 0, 1.0, ranges)

    Y = (X - mean) / ranges

    total_scatter = float(np.sum(Y ** 2))

    return mean, ranges, total_scatter


def normalized_squared_distances(X: FloatArray,indices: list[int],scales: FloatArray,reference: FloatArray,) -> FloatArray:
    X = np.asarray(X, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    X_subset = X[indices]

    diff = (X_subset - reference) / scales

    distances = np.sum(diff ** 2, axis=1)

    return distances


def cluster_centroid(X: FloatArray,indices: list[int]) -> FloatArray:
    
    X = np.asarray(X, dtype=np.float64)

    # selecionar pontos do cluster
    cluster_points = X[indices]

    # calcular média por feature
    centroid = np.mean(cluster_points, axis=0)

    return centroid



def separate_cluster(X: FloatArray,indices: list[int],scales: FloatArray,a: FloatArray,b: FloatArray) -> list[int]:

    dist_to_a = normalized_squared_distances(
        X=X,
        indices=indices,
        scales=scales,
        reference=a
    )

    dist_to_b = normalized_squared_distances(
        X=X,
        indices=indices,
        scales=scales,
        reference=b
    )

    selected = [
        idx for idx, da, db in zip(indices, dist_to_a, dist_to_b)
        if da < db
    ]

    return sorted(selected)

def extract_anomalous_cluster(
    X: FloatArray,
    indices: list[int],
    scales: FloatArray,
    mean: FloatArray,
    initial_centroid: FloatArray,
    seed_index: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> tuple[list[int], FloatArray]:

    X = np.asarray(X, dtype=np.float64)

    current_centroid = np.asarray(initial_centroid, dtype=np.float64).copy()
    previous_cluster: set[int] = set()
    new_cluster = [seed_index]

    for _ in range(max_iter):
        new_cluster = separate_cluster(
            X=X,
            indices=indices,
            scales=scales,
            a=current_centroid,
            b=mean,
        )

        if len(new_cluster) == 0:
            new_cluster = [seed_index]

        new_centroid = cluster_centroid(X, new_cluster)

        membership_stable = set(new_cluster) == previous_cluster
        centroid_shift = np.linalg.norm(new_centroid - current_centroid)

        if membership_stable or centroid_shift <= tol:
            return sorted(new_cluster), new_centroid

        current_centroid = new_centroid
        previous_cluster = set(new_cluster)

    final_centroid = cluster_centroid(X, new_cluster)
    return sorted(new_cluster), final_centroid

def ikmeans_initialize(
    X: FloatArray,
    min_cluster_size: int,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    use_unit_ranges: bool = False,
) -> tuple[list[APCluster], FloatArray]:

    X = np.asarray(X, dtype=np.float64)

    mean, scales, total_scatter = compute_feature_statistics(
        X,
        use_unit_ranges=use_unit_ranges
    )

    scales = np.where(scales == 0, 1.0, scales)

    remaining = list(range(X.shape[0]))
    clusters: list[APCluster] = []

    while len(remaining) > 0:

        dists_to_mean = normalized_squared_distances(
            X=X,
            indices=remaining,
            scales=scales,
            reference=mean
        )

        seed_idx = remaining[int(np.argmax(dists_to_mean))]
        seed = X[seed_idx]

        current_cluster, centroid_raw = extract_anomalous_cluster(
            X=X,
            indices=remaining,
            scales=scales,
            mean=mean,
            initial_centroid=seed,
            seed_index=seed_idx,
            tol=tol,
            max_iter=max_iter,
        )

        centroid_std = (centroid_raw - mean) / scales

        if total_scatter > 0:
            scatter_pct = (
                100.0
                * len(current_cluster)
                * np.sum(centroid_std ** 2)
                / total_scatter
            )
        else:
            scatter_pct = 0.0

        cluster = APCluster(
            indices=sorted(current_cluster),
            centroid_raw=centroid_raw,
            centroid_std=centroid_std,
            size=len(current_cluster),
            scatter_pct=float(scatter_pct),
        )

        clusters.append(cluster)

        current_set = set(current_cluster)
        remaining = [i for i in remaining if i not in current_set]

    retained_clusters = [c for c in clusters if c.size >= min_cluster_size]

    if len(retained_clusters) == 0:
        raise ValueError("No anomalous cluster satisfies the minimum size.")

    init_centroids = np.array(
        [c.centroid_raw for c in retained_clusters],
        dtype=np.float64
    )

    return retained_clusters, init_centroids

def test():
    X = np.loadtxt("C:/Users/user/Desktop/un/ikmeans/ikmeans/iris.dat", dtype=np.float64)

    ap_clusters, init_centroids = ikmeans_initialize(
        X,
        min_cluster_size=10,
        use_unit_ranges=True
    )

    labels, final_centroids, history = kmeans_scratch(
        X,
        initial_centroids=init_centroids,
        max_iter=300
    )

    print("Final clusters:", np.unique(labels, return_counts=True))
    print("Final centroids:")
    print(np.round(final_centroids, 2))

#test()

def explore_ikmeans_min_cluster_sizes(
    X,
    min_cluster_sizes,
    max_iter_ap=10_000,
    max_iter_kmeans=300,
    tol=1e-12,
    use_unit_ranges=True,
):
    rows = []

    for min_cluster_size in min_cluster_sizes:

        ap_clusters, init_centroids = ikmeans_initialize(
            X=X,
            min_cluster_size=min_cluster_size,
            tol=tol,
            max_iter=max_iter_ap,
            use_unit_ranges=use_unit_ranges,
        )

        labels, centroids, history = kmeans_scratch(
            X,
            initial_centroids=init_centroids,
            max_iter=max_iter_kmeans,
        )

        metrics = evaluate_kmeans_run(X, labels, centroids, history)

        rows.append({
            "method": "ikmeans",
            "min_cluster_size": min_cluster_size,
            "K": len(ap_clusters),
            "silhouette": metrics["silhouette"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "davies_bouldin": metrics["davies_bouldin"],
            "n_iterations": metrics["n_iterations"],
            "converged": metrics["converged"],
            "ap_cluster_sizes": [c.size for c in ap_clusters],
            "ap_scatter_pct": [c.scatter_pct for c in ap_clusters],

            # em memória, não para CSV
            "labels": labels,
            "centroids": centroids,
            "history": history,
            "cluster_sizes": metrics["cluster_sizes"],
            "ap_clusters": ap_clusters,
        })

    all_runs = pd.DataFrame(rows)

    export_cols = [
        "method",
        "min_cluster_size",
        "K",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "n_iterations",
        "converged",
        "ap_cluster_sizes",
        "ap_scatter_pct",
    ]

    all_runs_clean = all_runs[export_cols].copy()

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

    return all_runs, all_runs_clean, best_metric_runs

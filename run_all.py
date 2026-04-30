import os
from src.data_preparation import load_data,report_missingness,data_quality_snapshot
from src.data_preparation import run_preprocessing
from src.clustering import run_clustering,kmeans_scratch
from src.evaluation import evaluate_kmeans_run
from src.dataset_integrity import verify_dataset_integrity
import pandas as pd


base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, "data")
data_set_path = os.path.join(base_path, "data/hotel_bookings_course_release_v1.csv")



def explore_k_values_kmeans(X,K_values,M,max_iter=100,base_seed=1000):
    rows = []

    for K in K_values:
        for run in range(1, M + 1):
            seed = base_seed + 100 * K + run

            labels, centroids, history = kmeans_scratch(X,K=K,max_iter=max_iter,seed=seed)

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


def save_kmeans_tables(all_runs_clean,summary_by_k,best_metric_runs,output_dir="tables"):
   

    os.makedirs(output_dir, exist_ok=True)

    all_runs_clean.to_csv(
        os.path.join(output_dir, "kmeans_all_runs_clean.csv"),
        index=False
    )

    summary_by_k.to_csv(
        os.path.join(output_dir, "kmeans_summary_by_k.csv")
    )

    best_metric_runs.to_csv(
        os.path.join(output_dir, "kmeans_best_metric_runs.csv"),
        index=False
    )

def main():
    ok, details = verify_dataset_integrity(data_dir)

    if not ok:
        raise ValueError("Dataset integrity check failed. Check the files in the data folder.")
    df = pd.read_csv(data_set_path)
    X_std=run_preprocessing(df)
    #run_clustering(df)
    #run_evaluation()
    labels, centroids, history = kmeans_scratch(X_std, K=5, max_iter=100, seed=42)

    metrics = evaluate_kmeans_run(X_std, labels, centroids, history)

    metrics
    print(pd.DataFrame([metrics]).drop(columns=["cluster_sizes"]))

    all_runs_clean, summary_by_k, best_metric_runs = explore_k_values_kmeans(X=X_std,K_values=range(2, 4), M=3,max_iter=100,base_seed=1000)

    print("\n=== All Runs Clean ===")
    print(all_runs_clean.to_string(index=False))

    print("\n=== Summary by K ===")
    print(summary_by_k)

    print("\n=== Best Metric Runs ===")
    print(best_metric_runs.to_string(index=False))

    save_kmeans_tables(all_runs_clean, summary_by_k, best_metric_runs)

if __name__ == "__main__":
    main()
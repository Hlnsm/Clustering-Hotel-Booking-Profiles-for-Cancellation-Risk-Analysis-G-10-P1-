import os
import time
from datetime import datetime
from src.data_preparation import load_data,data_quality_snapshot
from src.data_preparation import run_preprocessing
from src.clustering import run_clustering,kmeans_scratch,explore_k_values_kmeans
from src.evaluation import evaluate_kmeans_run,compute_ari_by_k,centroid_profile
from src.dataset_integrity import verify_dataset_integrity
from src.ik_means import ikmeans_initialize,explore_ikmeans_min_cluster_sizes
from src.cancellation_profile import compute_and_save_cancellation_profile
from src.gmm_clustering import explore_k_values_gmm
import pandas as pd
from src.experiment_tables import (
    save_kmeans_exploration,
    save_ikmeans_exploration,
    save_experiments_csv,
    save_gmm_exploration,
)





base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, "data")
data_set_path = os.path.join(base_path, "data/hotel_bookings_course_release_v1.csv")



def main():
    pipeline_start = time.perf_counter()
    start_time = datetime.now().isoformat(timespec="seconds")

    ok, details = verify_dataset_integrity(data_dir)

    if not ok:
        raise ValueError("Dataset integrity check failed. Check the files in the data folder.")
    df = pd.read_csv(data_set_path)
    
    r_euclid_standard_noADR, r_euclid_robust_noADR, df_original,df_before_scaling=run_preprocessing(df)

    representations = {
    "R0-Euclid-standard-noADR": r_euclid_standard_noADR,
    "R1-Euclid-robust-noADR": r_euclid_robust_noADR,
    }

    experiment_logs = []

    for representation_id, X in representations.items():
        
        all_runs_clean, summary_by_k, best_metric_runs = explore_k_values_kmeans(
            X=X,
            K_values=range(2, 9),
            M=10,
            max_iter=100,
            base_seed=1000,
        )

        kmeans_experiments = save_kmeans_exploration(
            all_runs_clean=all_runs_clean,
            summary_by_k=summary_by_k,
            best_metric_runs=best_metric_runs,
            representation_id=representation_id,
        )

        experiment_logs.append(kmeans_experiments)

        ik_all_runs, ik_all_runs_clean, ik_best_metric_runs = explore_ikmeans_min_cluster_sizes(
            X=X,
            #1000, 1500, 2000, 3000, 5000, 8000, 9000, 11000
            min_cluster_sizes=[1000, 1500, 2000, 3000, 5000, 8000, 9000, 11000],
            max_iter_ap=10_000,
            max_iter_kmeans=300,
            tol=1e-12,
            use_unit_ranges=True,
        )

        ikmeans_experiments = save_ikmeans_exploration(
            all_runs_clean=ik_all_runs_clean,
            best_metric_runs=ik_best_metric_runs,
            representation_id=representation_id,
        )

        experiment_logs.append(ikmeans_experiments)
        
        gmm_all_runs, gmm_all_runs_clean, gmm_summary_by_k, gmm_best_metric_runs = explore_k_values_gmm(
            X=X,
            K_values=range(2, 9),
            M=5,
            covariance_types=("diag","tied"),
            max_iter=300,
            tol=1e-3,
            reg_covar=1e-6,
            base_seed=3000,
        )

        gmm_experiments = save_gmm_exploration(
            all_runs_clean=gmm_all_runs_clean,
            summary_by_k=gmm_summary_by_k,
            best_metric_runs=gmm_best_metric_runs,
            representation_id=representation_id,
        )

        experiment_logs.append(gmm_experiments)

    save_experiments_csv(experiment_logs)

    pipeline_end = time.perf_counter()
    end_time = datetime.now().isoformat(timespec="seconds")

    runtime_table = pd.DataFrame([
        {
            "run": "full_pipeline",
            "start_time": start_time,
            "end_time": end_time,
            "runtime_seconds": round(pipeline_end - pipeline_start, 4),
            "runtime_minutes": round((pipeline_end - pipeline_start) / 60, 4),
        }
    ])

    os.makedirs("tables", exist_ok=True)
    runtime_table.to_csv("tables/runtime_full_pipeline.csv", index=False)
"""

    #strandard



    #melhor run do k-means para avaliação
    labels,centroids_k5,history=kmeans_scratch(r_euclid_standard_noADR, K=5, max_iter=100, seed=1504)



    X_std,df_original, df_before_scaling=run_preprocessing(df)
    #run_clustering(df)
    #run_evaluation()
    if len(X_std) != len(df_original):
        raise ValueError(
            f"Row mismatch: X_std has {len(X_std)} rows but df_original has {len(df_original)} rows."
        )
    
    labels,centroids_k5,history=kmeans_scratch(X_std, K=5, max_iter=100, seed=1504)

    compute_and_save_cancellation_profile(
    df_original=df_original,
    labels=labels,
    method="kmeans",
    representation_id="R0-standard-noADR_kmeans_K5_seed1504",
    filename="cancellation_profile_kmeans_K5_seed1504.csv",
)

    #profile_tables_k3 = centroid_profile(X_std,centroids_k5,top_n=12,representation_id="kmeans_K5_seed1504")

    #save_profile_tables(profile_tables_k3, K=5, seed=1504)


    #all_runs_clean, summary_by_k, best_metric_runs = explore_k_values_kmeans(X=X_std,K_values=range(2, 9), M=10,max_iter=100,base_seed=1000)


    #save_kmeans_tables(all_runs_clean, summary_by_k, best_metric_runs)
    

    #ik_all_runs, ik_all_runs_clean, ik_best_metric_runs = explore_ikmeans_min_cluster_sizes(X=X_std,
    #min_cluster_sizes=[1000, 1500, 2000, 3000, 5000, 8000, 9000, 11000],
    #max_iter_ap=10_000,
    #max_iter_kmeans=300,
    #tol=1e-12,
    #use_unit_ranges=True)

    #save_ikmeans_tables(ik_all_runs_clean,ik_best_metric_runs)
"""



if __name__ == "__main__":
    main()

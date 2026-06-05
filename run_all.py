import os
import sys
import time
from datetime import datetime
from sklearn.mixture import GaussianMixture
from src.data_preparation import load_data,data_quality_snapshot
from src.data_preparation import run_preprocessing
from src.clustering import run_clustering,kmeans_scratch,explore_k_values_kmeans
from src.evaluation import evaluate_kmeans_run,compute_ari_by_k,centroid_profile,save_profile_tables
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
cluster_profiles_dir = "tables/clusterProfiles"


def get_run_config(mode):
    if mode == "fast":
        return {
            "run_name": "fast_check",
            "kmeans_K_values": range(2, 4),
            "kmeans_M": 2,
            "ikmeans_min_cluster_sizes": [3000, 5000],
            "gmm_K_values": range(2, 4),
            "gmm_M": 2,
            "run_posthoc": False,
            "tables_base_dir": os.path.join("tables", "fast_check"),
            "runtime_filename": "runtime_fast_check.csv",
        }

    if mode == "full":
        return {
            "run_name": "full_pipeline",
            "kmeans_K_values": range(2, 9),
            "kmeans_M": 10,
            "ikmeans_min_cluster_sizes": [1000, 1500, 2000, 3000, 5000, 8000, 9000, 11000],
            "gmm_K_values": range(2, 9),
            "gmm_M": 5,
            "run_posthoc": True,
            "tables_base_dir": "tables",
            "runtime_filename": "runtime_full_pipeline.csv",
        }

    raise ValueError("Run mode must be 'full' or 'fast'. Use: python run_all.py fast")


def get_run_mode():
    if len(sys.argv) == 1:
        return "full"

    if len(sys.argv) == 2:
        return sys.argv[1].strip().lower()

    raise ValueError("Too many arguments. Use: python run_all.py or python run_all.py fast")


def save_kmeans_posthoc_profile(
    X,
    df_original,
    representation_id,
    K,
    run,
    seed,
    top_n=12,):
    profile_id = f"{representation_id}_kmeans_K{K}_run{run}_seed{seed}"

    labels, centroids, history = kmeans_scratch(
        X,
        K=K,
        max_iter=100,
        seed=seed,)

    profile_tables = centroid_profile(
        X,
        centroids,
        top_n=top_n,
        representation_id=profile_id,)

    output_dir = save_profile_tables(
        profile_tables,
        profile_id=profile_id,
        output_base_dir=cluster_profiles_dir,)

    compute_and_save_cancellation_profile(
        df_original=df_original,
        labels=labels,
        method="kmeans",
        representation_id=profile_id,
        filename="cancellation_profile.csv",
        output_dir=output_dir,)

    return labels, centroids, history


def save_ikmeans_posthoc_profile(
    X,
    df_original,
    representation_id,
    min_cluster_size,
    expected_K=None,
    top_n=12,):
    ap_clusters, init_centroids = ikmeans_initialize(
        X=X,
        min_cluster_size=min_cluster_size,
        tol=1e-12,
        max_iter=10_000,
        use_unit_ranges=True,)

    if expected_K is not None and len(ap_clusters) != expected_K:
        raise ValueError(
            f"Expected iK-means K={expected_K} for min_cluster_size={min_cluster_size}, "
            f"got K={len(ap_clusters)}."
        )

    profile_id = f"{representation_id}_ikmeans_minSize{min_cluster_size}_K{len(ap_clusters)}"

    labels, centroids, history = kmeans_scratch(
        X,
        initial_centroids=init_centroids,
        max_iter=300,)

    profile_tables = centroid_profile(
        X,
        centroids,
        top_n=top_n,
        representation_id=profile_id,)

    output_dir = save_profile_tables(
        profile_tables,
        profile_id=profile_id,
        output_base_dir=cluster_profiles_dir,)

    compute_and_save_cancellation_profile(
        df_original=df_original,
        labels=labels,
        method="ikmeans",
        representation_id=profile_id,
        filename="cancellation_profile.csv",
        output_dir=output_dir,)

    return labels, centroids, history


def save_gmm_posthoc_profile(
    X,
    df_original,
    representation_id,
    covariance_type,
    K,
    run,
    seed,
    top_n=12,):
    profile_id = f"{representation_id}_gmm_{covariance_type}_K{K}_run{run}_seed{seed}"

    model = GaussianMixture(
        n_components=K,
        covariance_type=covariance_type,
        n_init=1,
        max_iter=300,
        tol=1e-3,
        reg_covar=1e-6,
        random_state=seed,)

    model.fit(X)
    labels = model.predict(X)

    centroids = (
        X
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
        .reindex(range(K))
        .to_numpy()
    )

    profile_tables = centroid_profile(
        X,
        centroids,
        top_n=top_n,
        representation_id=profile_id,)

    output_dir = save_profile_tables(
        profile_tables,
        profile_id=profile_id,
        output_base_dir=cluster_profiles_dir,)

    compute_and_save_cancellation_profile(
        df_original=df_original,
        labels=labels,
        method="gmm",
        representation_id=profile_id,
        filename="cancellation_profile.csv",
        output_dir=output_dir,)

    return labels, centroids, model


def main():
    pipeline_start = time.perf_counter()
    start_time = datetime.now().isoformat(timespec="seconds")
    mode = get_run_mode()
    config = get_run_config(mode)
    print(f"Running pipeline mode: {mode}")

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
            K_values=config["kmeans_K_values"],
            M=config["kmeans_M"],
            max_iter=100,
            base_seed=1000,
        )

        kmeans_experiments = save_kmeans_exploration(
            all_runs_clean=all_runs_clean,
            summary_by_k=summary_by_k,
            best_metric_runs=best_metric_runs,
            representation_id=representation_id,
            base_dir=config["tables_base_dir"],
        )

        if mode == "fast":
            kmeans_experiments["parameters"] = "K_grid=2-3;M=2;max_iter=100;base_seed=1000"
            kmeans_experiments["notes"] = "FAST CHECK ONLY: reduced k-means grid for pipeline/schema validation"

        experiment_logs.append(kmeans_experiments)

        ik_all_runs, ik_all_runs_clean, ik_best_metric_runs = explore_ikmeans_min_cluster_sizes(
            X=X,
            min_cluster_sizes=config["ikmeans_min_cluster_sizes"],
            max_iter_ap=10_000,
            max_iter_kmeans=300,
            tol=1e-12,
            use_unit_ranges=True,
        )

        ikmeans_experiments = save_ikmeans_exploration(
            all_runs_clean=ik_all_runs_clean,
            best_metric_runs=ik_best_metric_runs,
            representation_id=representation_id,
            base_dir=config["tables_base_dir"],
        )

        if mode == "fast":
            ikmeans_experiments["parameters"] = (
                "min_cluster_size_grid=[3000,5000];"
                "max_iter_ap=10000;max_iter_kmeans=300;tol=1e-12;use_unit_ranges=True"
            )
            ikmeans_experiments["notes"] = "FAST CHECK ONLY: reduced iK-means grid for pipeline/schema validation"

        experiment_logs.append(ikmeans_experiments)
        
        gmm_all_runs, gmm_all_runs_clean, gmm_summary_by_k, gmm_best_metric_runs = explore_k_values_gmm(
            X=X,
            K_values=config["gmm_K_values"],
            M=config["gmm_M"],
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
            base_dir=config["tables_base_dir"],
        )

        if mode == "fast":
            gmm_experiments["parameters"] = (
                gmm_experiments["parameters"]
                .str.replace("K_grid=2..8;", "K_grid=2..3;", regex=False)
                .str.replace("M=5;", "M=2;", regex=False)
            )
            gmm_experiments["notes"] = "FAST CHECK ONLY: reduced GMM grid for pipeline/schema validation"

        experiment_logs.append(gmm_experiments)

    save_experiments_csv(
        experiment_logs,
        base_dir=config["tables_base_dir"],
    )

    if config["run_posthoc"]:
        # Post-hoc profiling for the selected final runs.
        save_kmeans_posthoc_profile(
            X=r_euclid_standard_noADR,
            df_original=df_original,
            representation_id="R0-standard-noADR",
            K=7,
            run=9,
            seed=1709,
            top_n=12,)

        save_ikmeans_posthoc_profile(
            X=r_euclid_standard_noADR,
            df_original=df_original,
            representation_id="R0-standard-noADR",
            min_cluster_size=3000,
            expected_K=8,
            top_n=12,)

        save_gmm_posthoc_profile(
            X=r_euclid_standard_noADR,
            df_original=df_original,
            representation_id="R0-standard-noADR",
            covariance_type="tied",
            K=2,
            run=3,
            seed=3203,
            top_n=12,)

        save_kmeans_posthoc_profile(
            X=r_euclid_robust_noADR,
            df_original=df_original,
            representation_id="R1-robust-noADR",
            K=5,
            run=10,
            seed=1510,
            top_n=12,)

        save_ikmeans_posthoc_profile(
            X=r_euclid_robust_noADR,
            df_original=df_original,
            representation_id="R1-robust-noADR",
            min_cluster_size=3000,
            expected_K=9,
            top_n=12,)

        save_gmm_posthoc_profile(
            X=r_euclid_robust_noADR,
            df_original=df_original,
            representation_id="R1-robust-noADR",
            covariance_type="tied",
            K=8,
            run=1,
            seed=3801,
            top_n=12,)
    else:
        print("Fast mode: skipping selected-run post-hoc profiling.")

    pipeline_end = time.perf_counter()
    end_time = datetime.now().isoformat(timespec="seconds")

    runtime_table = pd.DataFrame([
        {
            "run": config["run_name"],
            "start_time": start_time,
            "end_time": end_time,
            "runtime_seconds": round(pipeline_end - pipeline_start, 4),
            "runtime_minutes": round((pipeline_end - pipeline_start) / 60, 4),
        }
    ])

    os.makedirs(config["tables_base_dir"], exist_ok=True)
    runtime_table.to_csv(
        os.path.join(config["tables_base_dir"], config["runtime_filename"]),
        index=False,
    )





if __name__ == "__main__":
    main()

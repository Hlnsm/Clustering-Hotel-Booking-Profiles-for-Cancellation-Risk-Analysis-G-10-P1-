import os
from datetime import datetime

import pandas as pd


EXPERIMENT_COLUMNS = [
    "timestamp",
    "run_id",
    "representation_id",
    "preprocessing_variant",
    "method",
    "K",
    "run",
    "seed",
    "min_cluster_size",
    "parameters",
    "sample_rule",
    "silhouette",
    "calinski_harabasz",
    "davies_bouldin",
    "n_iterations",
    "converged",
    "notes",
]


def safe_name(text: str) -> str:
    return (
        str(text)
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def preprocessing_variant(representation_id: str) -> str:
    rep = representation_id.lower()

    if "robust" in rep:
        return "robust"

    if "standard" in rep:
        return "standard"

    return "unknown"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def representation_output_dir(
    representation_id: str,
    base_dir: str = "tables",
) -> str:
    output_dir = os.path.join(base_dir, safe_name(representation_id))
    ensure_dir(output_dir)
    return output_dir




def normalise_experiment_table(
    runs: pd.DataFrame,
    representation_id: str,
    method: str,
    parameters: str = "",
    sample_rule: str = "full_data",
    notes: str = "",
) -> pd.DataFrame:
    

    experiments = runs.copy()

    experiments["timestamp"] = datetime.now().isoformat(timespec="seconds")
    experiments["representation_id"] = representation_id
    experiments["preprocessing_variant"] = preprocessing_variant(representation_id)
    experiments["method"] = method
    if "parameters" not in experiments.columns or parameters is not None:
        experiments["parameters"] = parameters
    experiments["sample_rule"] = sample_rule
    experiments["notes"] = notes

    if "run" not in experiments.columns:
        experiments["run"] = None

    if "seed" not in experiments.columns:
        experiments["seed"] = None

    if "min_cluster_size" not in experiments.columns:
        experiments["min_cluster_size"] = None

    diagnostics_cols = [
        col for col in [
            "ap_cluster_sizes",
            "ap_scatter_pct",
            "cluster_sizes",
        ]
        if col in experiments.columns
    ]


    for col in EXPERIMENT_COLUMNS:
        if col not in experiments.columns:
            experiments[col] = None

    experiments["run_id"] = experiments.apply(
        lambda row: build_run_id(row),
        axis=1,
    )

    experiments = experiments[EXPERIMENT_COLUMNS].copy()

    return experiments


def build_run_id(row: pd.Series) -> str:
    representation = safe_name(row.get("representation_id", "rep"))
    method = safe_name(row.get("method", "method"))
    k = row.get("K", "NA")
    run = row.get("run", None)
    seed = row.get("seed", None)
    min_cluster_size = row.get("min_cluster_size", None)
    covariance_type = row.get("covariance_type", None)

    parts = [
        representation,
        method,
        f"K{k}",
    ]

    if pd.notna(covariance_type):
        parts.append(f"cov{safe_name(covariance_type)}")

    if pd.notna(run):
        parts.append(f"run{int(run)}")

    if pd.notna(seed):
        parts.append(f"seed{int(seed)}")

    if pd.notna(min_cluster_size):
        parts.append(f"minSize{int(min_cluster_size)}")

    return "_".join(parts)


def save_kmeans_exploration(
    all_runs_clean: pd.DataFrame,
    summary_by_k: pd.DataFrame,
    best_metric_runs: pd.DataFrame,
    representation_id: str,
    base_dir: str = "tables"):
   

    output_dir = representation_output_dir(representation_id, base_dir)

    all_runs_clean.to_csv(os.path.join(output_dir, "kmeans_all_runs.csv"),index=False)

    summary_by_k.to_csv(os.path.join(output_dir, "kmeans_summary_by_k.csv"),index=False)

    best_metric_runs.to_csv(os.path.join(output_dir, "kmeans_best_metric_runs.csv"),index=False)

    experiments = normalise_experiment_table(
        runs=all_runs_clean,
        representation_id=representation_id,
        method="kmeans",
        parameters="K_grid=2-8;M=10;max_iter=100;base_seed=1000",
        sample_rule="full_data",
        notes="k-means baseline exploration")

    return experiments


def save_ikmeans_exploration(
    all_runs_clean: pd.DataFrame,
    best_metric_runs: pd.DataFrame,
    representation_id: str,
    base_dir: str = "tables"):
    

    output_dir = representation_output_dir(representation_id, base_dir)

    all_runs_clean.to_csv(
        os.path.join(output_dir, "ikmeans_all_runs.csv"),
        index=False)

    best_metric_runs.to_csv(os.path.join(output_dir, "ikmeans_best_metric_runs.csv"),index=False)

    experiments = normalise_experiment_table(
        runs=all_runs_clean,
        representation_id=representation_id,
        method="ikmeans",
        parameters=(
            "min_cluster_size_grid=[1000,1500,2000,3000,5000,8000,9000,11000];"
            "max_iter_ap=10000;max_iter_kmeans=300;tol=1e-12;use_unit_ranges=True"),
        sample_rule="full_data",
        notes="iK-means exploration")

    return experiments


def save_experiments_csv(
    experiment_tables: list[pd.DataFrame],
    base_dir: str = "tables",filename: str = "experiments.csv"):
    
    ensure_dir(base_dir)

    experiments = pd.concat(experiment_tables, ignore_index=True)

    output_path = os.path.join(base_dir, filename)

    experiments.to_csv(output_path, index=False)

    return experiments

def save_gmm_exploration(
    all_runs_clean: pd.DataFrame,
    summary_by_k: pd.DataFrame,
    best_metric_runs: pd.DataFrame,
    representation_id: str,
    base_dir: str = "tables",):

    output_dir = representation_output_dir(representation_id, base_dir)

    all_runs_clean.to_csv(
        os.path.join(output_dir, "gmm_all_runs.csv"),
        index=False,
    )

    summary_by_k.to_csv(
        os.path.join(output_dir, "gmm_summary_by_k.csv"),
        index=False,
    )

    best_metric_runs.to_csv(
        os.path.join(output_dir, "gmm_best_metric_runs.csv"),
        index=False,
    )

    gmm_parameters = all_runs_clean["covariance_type"].apply(
        lambda covariance_type: (
            "K_grid=2..8;"
            f"covariance_type={covariance_type};"
            "M=5;"
            "max_iter=300;"
            "tol=1e-3;"
            "reg_covar=1e-6;"
            "base_seed=3000;"
            "n_init=1"
        )
    )

    experiments = normalise_experiment_table(
        runs=all_runs_clean,
        representation_id=representation_id,
        method="gmm",
        parameters=gmm_parameters,
        sample_rule="full_data",
        notes="GMM/EM alternative family; AIC/BIC and membership diagnostics are stored in the GMM-specific result tables.",
    )

    return experiments

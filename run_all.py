import os
from src.data_preparation import load_data,report_missingness,data_quality_snapshot
from src.data_preparation import run_preprocessing
from src.clustering import run_clustering,kmeans_scratch
from src.evaluation import evaluate_kmeans_run
import pandas as pd


base_path = os.path.dirname(__file__)
data_set_path = os.path.join(base_path, "data/hotel_bookings_course_release_v1.csv")

def main():
    df = pd.read_csv(data_set_path)
    X_std=run_preprocessing(df)
    #run_clustering(df)
    #run_evaluation()
    labels, centroids, history = kmeans_scratch(X_std, K=5, max_iter=100, seed=42)

    metrics = evaluate_kmeans_run(X_std, labels, centroids, history)

    metrics
    print(pd.DataFrame([metrics]).drop(columns=["cluster_sizes"]))

if __name__ == "__main__":
    main()
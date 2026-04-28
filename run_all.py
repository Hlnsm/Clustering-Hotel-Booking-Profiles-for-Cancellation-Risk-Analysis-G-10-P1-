import os
from src.data_preparation import load_data,report_missingness,data_quality_snapshot
from src.data_preparation import run_preprocessing
from src.clustering import run_clustering
import pandas as pd


base_path = os.path.dirname(__file__)
data_set_path = os.path.join(base_path, "data/hotel_bookings_course_release_v1.csv")

def main():
    df = pd.read_csv(data_set_path)
    df=run_preprocessing(df)
    run_clustering(df)
    #run_evaluation()

if __name__ == "__main__":
    main()
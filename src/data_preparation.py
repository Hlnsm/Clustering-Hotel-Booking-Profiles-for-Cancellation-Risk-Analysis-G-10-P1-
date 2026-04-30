import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

figures_dir = "figures/dataPreparation"
os.makedirs(figures_dir, exist_ok=True)

tables_dir = "tables/dataPreparation"
os.makedirs(tables_dir, exist_ok=True)



def load_data(path):
    return pd.read_csv(path)

def drop_columns(df):
    
    
    cols_to_drop = [
        "is_canceled",
        "arrival_date_year",
        "arrival_date_day_of_month",
        "arrival_date_week_number",
        "reservation_status",
        "assigned_room_type",
        "reservation_status_date",
        "booking_changes",
        "days_in_waiting_list",
        #recomemdado pela prof possivel analize extra para ver se faz sentido por no clustering ou n 
        #######################
        "adr",
        "meal",
        "reserved_room_type",
        "total_of_special_requests",
        "required_car_parking_spaces",

        ######################
        "country", #alta carnalidade possivel para determinar os prefies dos clusters
        "agent",
        "company"
    ]
    return df.drop(columns=cols_to_drop, errors="ignore")


def data_quality_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    snap = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=False),                 
        "pct_unique": (df.nunique(dropna=False) / n * 100).round(2),
    }).sort_values(["pct_missing", "pct_unique"], ascending=False)
    
    n_dups=df.duplicated().sum()
    
    return n_dups, snap


def detect_outliers_iqr(df):
    num_df = df.select_dtypes(include=["int64", "float64"])
    results = []

    for col in num_df.columns:
        Q1 = num_df[col].quantile(0.25)
        Q3 = num_df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR <=0:
            continue
    
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        n_outliers = ((num_df[col] < lower) | (num_df[col] > upper)).sum()
        pct_outliers = (n_outliers / len(num_df)) * 100

        results.append({
            "feature": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "lower_bound": lower,
            "upper_bound": upper,
            "n_outliers": n_outliers,
            "pct_outliers": round(pct_outliers, 3)
        })

    outliers_df = pd.DataFrame(results).sort_values(by="pct_outliers", ascending=False)
    return outliers_df



def zscore_scale(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)                       
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma

def robust_scale_iqr(X: np.ndarray) -> np.ndarray:
    med = np.median(X, axis=0)
    q1 = np.quantile(X, 0.25, axis=0)
    q3 = np.quantile(X, 0.75, axis=0)
    iqr = np.where((q3 - q1) == 0, 1.0, (q3 - q1))
    return (X - med) / iqr


def feature_transformations(df):
    df = df.copy()

    df["stays_in_weekend_nights"] = df["stays_in_weekend_nights"].clip(upper=3)
    df["stays_in_week_nights"] = df["stays_in_week_nights"].clip(upper=9)

    df["kids"] = df["children"] + df["babies"]
    df = df.drop(columns=["children", "babies"])

    df["has_previous_cancellation"] = (df["previous_cancellations"] > 0).astype(int)
    df = df.drop(columns=["previous_cancellations"])

    df["previous_bookings_not_canceled"] = np.log1p(df["previous_bookings_not_canceled"])
    df["lead_time"] = np.log1p(df["lead_time"])

    
    df["market_segment"] = df["market_segment"].replace({"Complementary": "Other","Aviation": "Other"})
    df["distribution_channel"] = df["distribution_channel"].replace({"GDS": "Other","Undefined": "Other"})

    return df

def generate_histograms(df, figures_dir):
    
    os.makedirs(figures_dir, exist_ok=True)
    histograms = {}

    for col in df.columns:
        plt.figure(figsize=(6, 4))

        if pd.api.types.is_numeric_dtype(df[col]):

            n_unique = df[col].nunique(dropna=True)

            if n_unique < 50:
                df[col].value_counts(dropna=False).sort_index().plot(kind="bar")
                plt.title(f"{col} (discrete numeric)")

            else:
                values = df[col].dropna().values

                hist, bin_edges = np.histogram(values, bins=30)

                histograms[col] = {
                    "hist": hist,
                    "bins": bin_edges
                }

                plt.bar(
                    bin_edges[:-1],
                    hist,
                    width=np.diff(bin_edges),
                    edgecolor="black",
                    align="edge"
                )

                plt.title(f"{col} (Bins)")

        else:
            df[col].value_counts().head(20).plot(kind="bar")
            plt.title(f"{col} (categorical)")

        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_col = col.replace(" ", "_")
        plt.savefig(f"{figures_dir}/{safe_col}.png")
        plt.close()

    return histograms

def categorical_counts_check(df):
    cat_cols = df.select_dtypes(include=["object"]).columns
    summaries = []

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        pct = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)

        summary = pd.DataFrame({
            "feature": col,
            "category": counts.index.astype(str),
            "count": counts.values,
            "percentage": pct.values
        })

        summaries.append(summary)

    if summaries:
        return pd.concat(summaries, ignore_index=True)

    return pd.DataFrame(columns=["feature", "category", "count", "percentage"])

def run_preprocessing(df):
    df_original = df.copy()

    n_dups_original, snap_data_set_original = data_quality_snapshot(df_original)
    snap_data_set_original.to_csv(f"{tables_dir}/01_snap_original_dataset.csv")

    duplicates_original = pd.DataFrame({
        "stage": ["original_dataset"],
        "n_duplicates": [n_dups_original]
    })
    duplicates_original.to_csv(f"{tables_dir}/01_duplicates_original_dataset.csv", index=False)

    df = drop_columns(df)

    df = df.dropna(subset=["children"])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    n_dups_after_cleaning, snap_after_cleaning = data_quality_snapshot(df)
    snap_after_cleaning.to_csv(f"{tables_dir}/02_snap_after_drops_and_cleaning.csv")

    duplicates_after_cleaning = pd.DataFrame({
        "stage": ["after_drops_and_cleaning"],
        "n_duplicates": [n_dups_after_cleaning]
    })
    
    duplicates_after_cleaning.to_csv(f"{tables_dir}/02_duplicates_after_drops_and_cleaning.csv", index=False)

    generate_histograms(df, f"{figures_dir}/01_original_histo")

    outliers_before = detect_outliers_iqr(df)
    outliers_before.to_csv(f"{tables_dir}/02_outliers_after_drops_and_cleaning.csv", index=False)

    df = feature_transformations(df)

    generate_histograms(df, f"{figures_dir}/02_after_feature_transformations_histo")

    df_before_scaling = df.copy()

    n_dups_after_transformations, snap_after_transformations = data_quality_snapshot(df_before_scaling)
    snap_after_transformations.to_csv(f"{tables_dir}/03_snapshot_after_feature_transformations.csv")

    numerical_cols = [
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "kids",
        "previous_bookings_not_canceled",
        "lead_time"
    ]

    binary_cols = [
        "has_previous_cancellation"
    ]

    df_numeric = df[numerical_cols]

    df_standard = pd.DataFrame(
        zscore_scale(df_numeric.values),
        columns=numerical_cols,
        index=df.index
    )

    df_robust = pd.DataFrame(
        robust_scale_iqr(df_numeric.values),
        columns=numerical_cols,
        index=df.index
    )

    cat_cols = df.select_dtypes(include=["object"]).columns
    df_one_hot = pd.get_dummies(df[cat_cols], prefix=cat_cols, dtype=int)

    df = pd.concat([df_standard, df[binary_cols], df_one_hot], axis=1)

    n_dups, snap_full_preprocessing = data_quality_snapshot(df)
    snap_full_preprocessing.to_csv(f"{tables_dir}/04_snap_final_representation.csv")

    return df, df_original, df_before_scaling
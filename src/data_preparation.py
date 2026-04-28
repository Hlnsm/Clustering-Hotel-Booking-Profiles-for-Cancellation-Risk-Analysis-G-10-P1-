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
    # as grandes questões se vale apena pro coisas complementares sobre o boooking como tipo :
    
    cols_to_drop = [
        "is_canceled",
        "arrival_date_year",
        "arrival_date_day_of_month",
        "arrival_date_week_number",
        "reservation_status",
        "assigned_room_type",
        "reservation_status_date",
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

def report_missingness(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    report = pd.DataFrame({
        "missing_count": missing,
        "missing_%": missing_pct
    }).sort_values(by="missing_%", ascending=False)

    print(report)
    return report

def data_quality_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    snap = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(dropna=False),                 # NaN counts as a distinct value in n_unique (dropna=False).
        "pct_unique": (df.nunique(dropna=False) / n * 100).round(2),
    }).sort_values(["pct_missing", "pct_unique"], ascending=False)
    
    n_dups=df.duplicated().sum()
    print("num of dupes ",n_dups)
    print(snap)
    return snap


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
#melhorar este funçao antes da entrega coisas com as colunas estarem no meio da funçao e tb os histogramas e salvar os df
def run_preprocessing(df):
    df = drop_columns(df)
    snap = data_quality_snapshot(df)
    snap.to_csv(f"{tables_dir}/snapshot.csv")
    df = df.dropna(subset=["children"])
    #fazer isto?
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    histograms = {}

    #fazer funçao unica isto 
    for col in df.columns:
        plt.figure(figsize=(6,4))

        # verificar tipo da coluna
        if pd.api.types.is_numeric_dtype(df[col]):

            n_unique = df[col].nunique(dropna=True)
            #se for menos q 20 val unicos usamos bar plot
            if n_unique < 50:
                df[col].value_counts(dropna=False).sort_index().plot(kind="bar")

                plt.title(f"{col} (discrete numeric)")
            #assima de 20 usamos histogramas com bins
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

                plt.title(f"{col} (continuous)")

        else:
            # categóricas bar plot
            df[col].value_counts().head(20).plot(kind="bar")
            plt.title(f"{col} (categorical)")

        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.xticks(rotation=45)
        plt.tight_layout()

        safe_col = col.replace(" ", "_")
        plt.savefig(f"{figures_dir}/{safe_col}.png")
        plt.close()

    #print(detect_outliers_iqr(df))


  

    numerical_cols = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list"
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

    #print(df_standard.describe())

    #print(df_robust.describe())

    #one hot encoding
    cat_cols = df.select_dtypes(include=["object"]).columns
    df_one_hot=pd.get_dummies(df[cat_cols],prefix=cat_cols,dtype=int)
    df=pd.concat([df_standard,df_one_hot],axis=1)


    #o que falta fazer guardar o std vs robost df em pastas e fazer todo tendo em conta essa comparaçao 

    #print(df)
    data_quality_snapshot(df)

    return df

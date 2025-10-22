

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import joblib
import mlflow
import mlflow.sklearn

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta

@task(retries=2, retry_delay_seconds=5, cache_key_fn=task_input_hash, cache_expiration=timedelta(days=1))
def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Data loaded. Shape: {df.shape}")
    return df


@task
def perform_eda(df: pd.DataFrame):
    avg_price_location = df.groupby("location")["price"].mean().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12,6))
    sns.barplot(x=avg_price_location.index, y=avg_price_location.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Average Price (Lakhs)")
    plt.title("Top 10 Locations by Average House Price")
    plt.tight_layout()
    eda_path = "eda_top10_locations.png"
    plt.savefig(eda_path)
    plt.close()
    print(f"✅ EDA plot saved as '{eda_path}'")
    return eda_path


@task
def build_pipeline(df: pd.DataFrame, target: str = "price"):
    X = df.drop(columns=[target])
    y = df[target]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline, X, y, numerical_cols, categorical_cols


@task
def train_evaluate(pipeline, X, y, numerical_cols, categorical_cols):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Ahmedabad_House_Price_Prediction")
    with mlflow.start_run(run_name="Prefect_XGBoost_pipeline"):
        print("\n🚀 Training model...")
        pipeline.fit(X_train, y_train)
        print("✅ Model trained!")

        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        print(f"\n📉 Test RMSE: {rmse:.2f}")
        mlflow.log_metric("rmse", rmse)

        # Feature importance
        preprocessor = pipeline.named_steps['preprocessor']
        preprocessor.fit(X_train)
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        all_features = numerical_cols + list(cat_features)

        importances = pipeline.named_steps['regressor'].feature_importances_
        fi_df = pd.DataFrame({"feature": all_features, "importance": importances}).sort_values(by="importance", ascending=False)
        fi_csv = "feature_importance.csv"
        fi_df.to_csv(fi_csv, index=False)
        mlflow.log_artifact(fi_csv)
        print("\n🌟 Top 10 features:\n", fi_df.head(10))

        # Save pipeline
        output_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "house_price_pipeline_prefect.pkl")
        joblib.dump(pipeline, output_path)
        mlflow.log_artifact(output_path)
        print(f"\n✅ Pipeline saved at {output_path}")

    return rmse, fi_csv, output_path


@flow(name="Ahmedabad_House_Price_Flow")
def main_flow():
    data_path = os.path.join(os.getcwd(), "data", "ahmedabad_cleaned.csv")
    df = load_data(data_path)
    perform_eda(df)
    pipeline, X, y, num_cols, cat_cols = build_pipeline(df)
    train_evaluate(pipeline, X, y, num_cols, cat_cols)

# Run the flow
if __name__ == "__main__":
    main_flow()

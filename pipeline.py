import mlflow
import os
from src.pipeline_components import fetch_data, preprocess_data, train_model, evaluate_model

DATA_PATH = "data/raw_data.csv"
MODEL_PATH = "data/processed/model.pkl"
METRICS_PATH = "data/processed/metrics.json"

os.makedirs("data/processed", exist_ok=True)

# Start an MLflow run
mlflow.set_experiment("housing_pipeline")

with mlflow.start_run(run_name="run1"):

    # 1. Fetch data
    df = fetch_data(DATA_PATH)
    
    # 2. Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    
    # 3. Train model
    model = train_model(X_train, y_train, MODEL_PATH)
    mlflow.sklearn.log_model(model, "model")
    
    # 4. Evaluate model
    metrics = evaluate_model(MODEL_PATH, X_test, y_test, METRICS_PATH)
    
    # Log metrics to MLflow
    mlflow.log_metrics(metrics)
    
    print("Pipeline completed. Metrics:", metrics)

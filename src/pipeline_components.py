import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# ----------------------
# Data Extraction
# ----------------------
def fetch_data(file_path: str) -> pd.DataFrame:
    """Fetches dataset from local DVC-tracked CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist")
    return pd.read_csv(file_path)

# ----------------------
# Data Preprocessing
# ----------------------
def preprocess_data(df: pd.DataFrame):
    """Cleans, scales, and splits data into train/test sets."""
    X = df.drop("MedHouseVal", axis=1, errors="ignore")  # TARGET column in California dataset
    y = df["MedHouseVal"] if "MedHouseVal" in df.columns else df.iloc[:, -1]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, scaler

# ----------------------
# Model Training
# ----------------------
def train_model(X_train, y_train, model_path: str):
    """Trains RandomForestRegressor and saves the model artifact."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

# ----------------------
# Model Evaluation
# ----------------------
def evaluate_model(model_path: str, X_test, y_test, metrics_path: str):
    """Loads model, evaluates it, and saves metrics."""
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    
    print(f"Metrics saved to {metrics_path}")
    return metrics

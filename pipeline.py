import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# ----------------------
# Step 0: Paths
# ----------------------
FULL_DATA_PATH = "data/raw_data.csv"          # Your original full dataset
SMALL_DATA_PATH = "data/raw_data_small.csv"   # Small dataset for GitHub
MODEL_PATH = "data/processed/model_small.pkl"
METRICS_PATH = "data/processed/metrics_small.json"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# ----------------------
# Step 1: Create small dataset
# ----------------------
if not os.path.exists(FULL_DATA_PATH):
    raise FileNotFoundError(f"{FULL_DATA_PATH} not found!")

# Take 10% sample to reduce size
df_full = pd.read_csv(FULL_DATA_PATH)
df_small = df_full.sample(frac=0.1, random_state=42)
df_small.to_csv(SMALL_DATA_PATH, index=False)
print(f"Small dataset created: {SMALL_DATA_PATH}, shape: {df_small.shape}")

# ----------------------
# Step 2: Data Extraction
# ----------------------
df = pd.read_csv(SMALL_DATA_PATH)
print(f"Loaded small dataset, shape: {df.shape}")

# ----------------------
# Step 3: Data Preprocessing
# ----------------------
# Assume target column is 'MedHouseVal'; adjust if using another dataset
target_col = "MedHouseVal" if "MedHouseVal" in df.columns else df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Data split: Train {X_train.shape}, Test {X_test.shape}")

# ----------------------
# Step 4: Model Training
# ----------------------
# Smaller RandomForest to stay under 100MB
model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ----------------------
# Step 5: Model Evaluation
# ----------------------
y_pred = model.predict(X_test)
metrics = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

# Save metrics
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f)

print(f"Metrics saved to {METRICS_PATH}")
print(f"Pipeline completed successfully: MSE={metrics['mse']:.3f}, R2={metrics['r2']:.3f}")

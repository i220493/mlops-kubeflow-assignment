# MLOps Assignment: Housing Price Prediction Pipeline

## Project Overview

This project demonstrates an **end-to-end MLOps workflow** for predicting housing prices using a lightweight dataset.  
The workflow includes:

- Data extraction and preprocessing  
- Model training using a **Random Forest Regressor**  
- Model evaluation and metrics tracking  
- Data versioning with **DVC**  
- Experiment tracking with **MLflow**  

The pipeline is lightweight and GitHub-friendly (model size < 100MB) and can run locally or via **GitHub Actions**.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/i220493/mlops-kubeflow-assignment.git
cd mlops-kubeflow-assignment
2. Python Environment
Ensure you have Python 3.11+ installed.

Optional: create a virtual environment:

bash
Copy code
python -m venv venv
# Activate
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
3. Dataset
data/raw_data_small.csv is included in the repository.

If you have a larger dataset (data/raw_data.csv), the pipeline can automatically create a smaller dataset by sampling 10% of the data.

4. DVC Setup
DVC is initialized in the repository.

The small dataset is tracked locally with DVC:

bash
Copy code
dvc add data/raw_data_small.csv
git add data/raw_data_small.csv.dvc .gitignore
git commit -m "Track small dataset with DVC"
No remote storage is required for this small dataset.

Pipeline Walkthrough
1. Run the Pipeline
Execute the pipeline script:

bash
Copy code
python pipeline_small.py
The pipeline performs the following steps:

Load the small dataset (data/raw_data_small.csv)

Preprocess features (scaling and train/test split)

Train a smaller Random Forest Regressor

Save the trained model artifact: data/processed/model_small.pkl

Evaluate the model and save metrics: data/processed/metrics_small.json

2. Output
Model: data/processed/model_small.pkl

Metrics: data/processed/metrics_small.json

Metrics include:

mse → Mean Squared Error

r2 → R² score

3. MLflow Tracking
MLflow is used for experiment tracking.

By default, experiments are stored in the mlruns folder:

bash
Copy code
mlflow ui
Access the MLflow dashboard at http://127.0.0.1:5000 to view experiments.

Project Notes
This project demonstrates a GitHub-friendly MLOps workflow.

The pipeline can be automated with GitHub Actions, running the training pipeline on push or pull request.

Model artifacts are <100MB and safe to commit to GitHub.

DVC allows easy dataset versioning and reproducibility.


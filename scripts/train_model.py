import os
import glob
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Absolute paths inside container
FEATURE_DIR = "/opt/airflow/datamart/gold/feature_label_store"
MODEL_DIR = "/opt/airflow/model_bank"
MONITOR_DIR = "/opt/airflow/monitoring"
TRAIN_LOG_FILE = os.path.join(MONITOR_DIR, "training_accuracy_log.csv")


def create_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MONITOR_DIR, exist_ok=True)

def log_accuracy(acc, date_str):
    log_path = TRAIN_LOG_FILE
    entry = pd.DataFrame([{
        "date": date_str,
        "model": "logistic_regression",
        "accuracy": acc
    }])
    if os.path.exists(log_path):
        prev = pd.read_csv(log_path)
        entry = pd.concat([prev, entry], ignore_index=True)
    entry.to_csv(log_path, index=False)

def is_training_date(filename, start="2023_02_01", end="2024_02_01"):
    try:
        date_part = os.path.basename(filename).split("feature_label_")[1].replace(".parquet", "")
        file_date = datetime.strptime(date_part, "%Y_%m_%d")
        return datetime.strptime(start, "%Y_%m_%d") <= file_date <= datetime.strptime(end, "%Y_%m_%d")
    except Exception as e:
        print(f"Date parsing failed for {filename}: {e}")
        return False

def load_training_data():
    files = glob.glob(os.path.join(FEATURE_DIR, "feature_label_*.parquet"))
    training_files = [f for f in files if is_training_date(f)]

    print(f"Using {len(training_files)} training files.")
    if not training_files:
        raise ValueError("No valid training files found. Check path and file availability.")

    dfs = [pd.read_parquet(f) for f in training_files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def main():
    create_dirs()
    today = datetime.today().strftime('%Y-%m-%d')

    print("Loading and combining training data...")
    df = load_training_data()

    features = [
        'Interest_Rate', 'Outstanding_Debt', 'Credit_History_Age', 'Loan_Extent',
        'Loans_per_Credit_Item', 'Repayment_Ability', 'Debt_to_Salary',
        'EMI_to_Salary', 'Num_Fin_Pdts'
    ]
    target = 'label'

    df = df[features + [target]].dropna()
    X, y = df[features], df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    print("Saving model and scaler...")
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(model, f, protocol=4)

    with open(os.path.join(MODEL_DIR, "best_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f, protocol=4)

    print("Logging accuracy...")
    log_accuracy(acc, today)

    print("Training complete.")

if __name__ == "__main__":
    main()

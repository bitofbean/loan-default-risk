import os
import pandas as pd
from glob import glob
from sklearn.metrics import accuracy_score, classification_report

# --- Docker container paths ---
BASE_PATH = "/opt/airflow/datamart/gold"
PRED_DIR = os.path.join(BASE_PATH, "model_predictions")
LABEL_DIR = os.path.join(BASE_PATH, "feature_label_store")
MONITOR_DIR = "/opt/airflow/monitoring"
LOG_FILE = os.path.join(MONITOR_DIR, "accuracy_log.csv")

def log_accuracy(date_str, model, accuracy):
    os.makedirs(MONITOR_DIR, exist_ok=True)
    entry = pd.DataFrame([{
        "date": date_str,
        "model": model,
        "accuracy": accuracy
    }])
    if os.path.exists(LOG_FILE):
        prev = pd.read_csv(LOG_FILE)
        entry = pd.concat([prev, entry], ignore_index=True)
    entry.to_csv(LOG_FILE, index=False)
    print(f"Logged accuracy to {LOG_FILE}")

def evaluate(snapshot_date, model_name="logistic_regression"):
    date_fmt = snapshot_date.replace("-", "_")
    pred_file = os.path.join(PRED_DIR, f"predictions_{date_fmt}.csv")
    label_file = os.path.join(LABEL_DIR, f"feature_label_{date_fmt}.parquet")

    print(f"\nEvaluating snapshot: {snapshot_date}")
    print(f"Prediction file: {pred_file}")
    print(f"Label file: {label_file}")

    if not os.path.exists(pred_file):
        print(f"SKIP: Missing prediction file: {pred_file}")
        return
    if not os.path.exists(label_file):
        print(f"SKIP: Missing label file: {label_file}")
        return

    try:
        df_pred = pd.read_csv(pred_file)
        print(f"Loaded {len(df_pred)} predictions")
    except Exception as e:
        print(f"ERROR: Failed to load predictions: {e}")
        return

    try:
        # Handle both parquet files and directories
        if os.path.isdir(label_file):
            # If it's a directory (Spark partitioned format)
            df_label = pd.read_parquet(label_file)
        else:
            # If it's a single file
            df_label = pd.read_parquet(label_file)
        print(f"Loaded {len(df_label)} labels")
    except Exception as e:
        print(f"ERROR: Failed to load labels: {e}")
        return

    # Check if required columns exist
    if "Customer_ID" not in df_pred.columns:
        print("ERROR: 'Customer_ID' column missing from predictions")
        return
    if "predicted_label" not in df_pred.columns:
        print("ERROR: 'predicted_label' column missing from predictions")
        return
    if "Customer_ID" not in df_label.columns:
        print("ERROR: 'Customer_ID' column missing from labels")
        return
    if "label" not in df_label.columns:
        print("ERROR: 'label' column missing from labels")
        return

    df_merged = pd.merge(df_pred, df_label[["Customer_ID", "label"]], on="Customer_ID", how="inner")
    print(f"Merged dataset size: {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("ERROR: No matching Customer_IDs found between predictions and labels")
        return

    acc = accuracy_score(df_merged["label"], df_merged["predicted_label"])

    print(f"\nSnapshot: {snapshot_date} - Accuracy: {acc:.4f}")
    print(classification_report(df_merged["label"], df_merged["predicted_label"]))

    log_accuracy(snapshot_date, model_name, acc)

def debug_files():
    """Debug function to check what files exist"""
    print("DEBUG: Checking file locations...")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"PRED_DIR: {PRED_DIR}")
    print(f"LABEL_DIR: {LABEL_DIR}")
    print(f"MONITOR_DIR: {MONITOR_DIR}")
    
    print(f"\nPrediction directory exists: {os.path.exists(PRED_DIR)}")
    if os.path.exists(PRED_DIR):
        pred_files = os.listdir(PRED_DIR)
        print(f"Prediction files: {pred_files}")
    
    print(f"\nLabel directory exists: {os.path.exists(LABEL_DIR)}")
    if os.path.exists(LABEL_DIR):
        label_files = os.listdir(LABEL_DIR)
        print(f"Label files: {label_files}")

def main():
    print("Starting model monitoring...")
    
    # Check if directories exist first
    if not os.path.exists(PRED_DIR):
        print(f"ERROR: Prediction directory not found: {PRED_DIR}")
        print("Make sure you've run the inference script first.")
        debug_files()
        return
        
    if not os.path.exists(LABEL_DIR):
        print(f"ERROR: Label directory not found: {LABEL_DIR}")
        debug_files()
        return

    pred_files = sorted(glob(os.path.join(PRED_DIR, "predictions_*.csv")), reverse=True)[:3]
    if not pred_files:
        print("No prediction files found.")
        print("Available files in prediction directory:")
        for f in os.listdir(PRED_DIR):
            print(f"  - {f}")
        return

    print(f"Found {len(pred_files)} prediction files to evaluate:")
    for f in pred_files:
        print(f"  - {os.path.basename(f)}")

    for f in reversed(pred_files):  # oldest to newest
        snapshot_fmt = os.path.basename(f).split("predictions_")[1].replace(".csv", "")
        snapshot_date = snapshot_fmt.replace("_", "-")
        evaluate(snapshot_date)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor model performance")
    parser.add_argument("--debug", action="store_true", help="Debug file locations")
    args = parser.parse_args()
    
    if args.debug:
        debug_files()
    else:
        main()
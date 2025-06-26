import argparse
import os
import pandas as pd
import pickle
from datetime import datetime
from glob import glob
from pyspark.sql import SparkSession

# --- Define base paths (configurable) ---
def get_paths(feature_dir=None, model_dir=None):
    """Get paths, with fallbacks to common locations"""
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Try multiple common locations for feature store
    if feature_dir:
        feature_paths = [feature_dir]
    else:
        feature_paths = [
            "/opt/airflow/datamart/gold/feature_store",  # Docker volume mount
            "/opt/airflow/data/datamart/gold/feature_store",  # Alternative mount
            os.path.join(base_dir, "datamart", "gold", "feature_store"),
            "/opt/airflow/data/feature_store",
            os.path.join(base_dir, "data", "feature_store"),
            base_dir  # Last resort: current directory
        ]
    
    # Try multiple common locations for models
    if model_dir:
        model_paths = [model_dir]
    else:
        model_paths = [
            "/opt/airflow/model_bank",  # Docker volume mount
            "/opt/airflow/data/model_bank",  # Alternative mount
            os.path.join(base_dir, "model_bank"),
            "/opt/airflow/models",
            os.path.join(base_dir, "models"),
            base_dir  # Last resort: current directory
        ]
    
    # Find first existing feature directory
    feature_dir_final = None
    for path in feature_paths:
        if os.path.exists(path):
            feature_dir_final = path
            break
    
    # Find first existing model directory  
    model_dir_final = None
    for path in model_paths:
        if os.path.exists(path):
            model_dir_final = path
            break
    
    # Create output directory relative to feature directory
    if feature_dir_final:
        output_dir = os.path.join(os.path.dirname(feature_dir_final), "model_predictions")
    else:
        output_dir = os.path.join(base_dir, "outputs")
    
    return feature_dir_final, model_dir_final, output_dir

# --- Inference function ---
def run_inference(snapshotdate, modelname, feature_dir, model_dir, output_dir):
    snapshot_fmt = snapshotdate.replace("-", "_")
    model_path = os.path.join(model_dir, "best_model.pkl")
    scaler_path = os.path.join(model_dir, "best_scaler.pkl")
    feature_path = os.path.join(feature_dir, f"joined_features_{snapshot_fmt}.parquet")
    output_file = os.path.join(output_dir, f"predictions_{snapshot_fmt}.csv")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nDate: {snapshotdate}")
    print(f"Input path: {feature_path}")
    print(f"Output path: {output_file}")

    # --- Init Spark ---
    try:
        print("Starting Spark session...")
        spark = SparkSession.builder.appName("InferenceJob").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    except Exception as e:
        print(f"ERROR: Spark session error: {e}")
        return

    # --- Load model & scaler ---
    print("Loading model and scaler...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print("SUCCESS: Model and scaler loaded successfully.")
    except Exception as e:
        print(f"ERROR: Error loading model/scaler: {e}")
        spark.stop()
        return

    # --- Load parquet as pandas ---
    print("Reading and converting parquet to pandas...")
    try:
        sdf = spark.read.parquet(feature_path)
        pdf = sdf.toPandas()
        print("SUCCESS: Parquet file loaded into pandas DataFrame.")
    except Exception as e:
        print(f"ERROR: Error reading {feature_path}: {e}")
        spark.stop()
        return

    # --- Feature selection ---
    feature_cols = [
        'Interest_Rate', 'Outstanding_Debt', 'Credit_History_Age', 'Loan_Extent',
        'Loans_per_Credit_Item', 'Repayment_Ability', 'Debt_to_Salary',
        'EMI_to_Salary', 'Num_Fin_Pdts'
    ]

    print("Checking required features...")
    if not set(feature_cols).issubset(pdf.columns):
        print(f"ERROR: Missing required features in {feature_path}")
        print(f"Available columns: {list(pdf.columns)}")
        spark.stop()
        return
    print("SUCCESS: All required features are present.")

    # --- Run inference ---
    try:
        print("Running predictions...")
        X = scaler.transform(pdf[feature_cols])
        y_pred_label = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        print("SUCCESS: Predictions complete.")
    except Exception as e:
        print(f"ERROR: Error during prediction: {e}")
        spark.stop()
        return

    # --- Save results ---
    try:
        print("Saving predictions to CSV...")
        results = pdf[["Customer_ID"]].copy()
        results["snapshot_date"] = snapshotdate
        results["model_name"] = modelname
        results["predicted_label"] = y_pred_label
        results["predicted_probability"] = y_pred_proba
        results.to_csv(output_file, index=False)
        print(f"SUCCESS: Predictions saved to: {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to save predictions: {e}")
    
    spark.stop()

def debug_and_find_files():
    """Find data files and suggest paths"""
    print(f"\nSEARCHING FOR DATA FILES...")
    
    # Search for parquet files (both files and directories)
    print("Searching for parquet files...")
    parquet_files = []
    for root, dirs, files in os.walk('/opt/airflow'):
        # Check for parquet files
        for file in files:
            if file.endswith('.parquet') and 'joined_features' in file:
                full_path = os.path.join(root, file)
                parquet_files.append(full_path)
                print(f"   FOUND: {full_path}")
        # Check for parquet directories (Spark partitioned format)
        for dir_name in dirs:
            if dir_name.endswith('.parquet') and 'joined_features' in dir_name:
                full_path = os.path.join(root, dir_name)
                parquet_files.append(full_path)
                print(f"   FOUND: {full_path} (directory)")
    
    # Search for model files
    print("\nSearching for model files...")
    model_files = []
    for root, dirs, files in os.walk('/opt/airflow'):
        for file in files:
            if file.endswith('.pkl') and ('model' in file or 'scaler' in file):
                full_path = os.path.join(root, file)
                model_files.append(full_path)
                print(f"   FOUND: {full_path}")
    
    if parquet_files:
        suggested_feature_dir = os.path.dirname(parquet_files[0])
        print(f"\nSuggested feature directory: {suggested_feature_dir}")
    
    if model_files:
        suggested_model_dir = os.path.dirname(model_files[0])
        print(f"Suggested model directory: {suggested_model_dir}")
    
    return parquet_files, model_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--snapshotdate", type=str, help="Specific snapshot date (YYYY-MM-DD)")
    parser.add_argument("--latest_oot", type=int, help="Use latest N snapshot files (overrides snapshotdate)")
    parser.add_argument("--modelname", type=str, required=True, help="Model name")
    parser.add_argument("--feature_dir", type=str, help="Override feature directory path")
    parser.add_argument("--model_dir", type=str, help="Override model directory path")
    parser.add_argument("--search", action="store_true", help="Search for data files in container")
    args = parser.parse_args()

    # Search for files if requested
    if args.search:
        debug_and_find_files()
        exit()

    # Get paths
    feature_dir, model_dir, output_dir = get_paths(args.feature_dir, args.model_dir)
    
    print(f"Using paths:")
    print(f"   Feature dir: {feature_dir}")
    print(f"   Model dir: {model_dir}")
    print(f"   Output dir: {output_dir}")
    
    # Check if essential directories exist
    if not feature_dir:
        print("ERROR: Feature directory not found!")
        print("Use --search to find your data files")
        exit(1)
        
    if not model_dir:
        print("ERROR: Model directory not found!")
        print("Use --search to find your model files")
        exit(1)

    if args.latest_oot:
        # Look for both .parquet files and .parquet directories
        parquet_pattern = os.path.join(feature_dir, "joined_features_*.parquet")
        parquet_files = sorted(glob(parquet_pattern), reverse=True)
        
        # Also check for parquet directories (if files are empty)
        if not parquet_files:
            parquet_dirs = sorted([
                d for d in glob(parquet_pattern) 
                if os.path.isdir(d)
            ], reverse=True)
            parquet_files = parquet_dirs
        
        latest_files = parquet_files[:args.latest_oot]
        print(f"\nRunning inference on latest {args.latest_oot} files:")
        print(f"Matched {len(parquet_files)} total parquet files.")
        
        if not latest_files:
            print("ERROR: No parquet files found!")
            print("Use --search to find your data files")
            exit(1)
            
        print(f"Using these snapshot files:")
        for f in reversed(latest_files):
            print("   -", f)

        for f in reversed(latest_files):
            try:
                date_part = os.path.basename(f).split("joined_features_")[1].replace(".parquet", "")
                snapshotdate = datetime.strptime(date_part, "%Y_%m_%d").strftime("%Y-%m-%d")
                print(f"\nCalling inference for: {snapshotdate}")
                run_inference(snapshotdate, args.modelname, feature_dir, model_dir, output_dir)
            except Exception as e:
                print(f"ERROR: Skipped file {f}: {e}")
                
    elif args.snapshotdate:
        run_inference(args.snapshotdate, args.modelname, feature_dir, model_dir, output_dir)
    else:
        print("ERROR: Please provide either --snapshotdate or --latest_oot")
        print("Use --search to find your data files first")
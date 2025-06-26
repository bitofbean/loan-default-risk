from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

# --- DAG Defaults ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- DAG Definition ---
with DAG(
    'loan_eligibility_pipeline',
    default_args=default_args,
    description='Monthly loan eligibility pipeline with training, inference, and monitoring',
    schedule_interval='0 0 1 * *',  # At 00:00 on the 1st day of each month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # --- Label Store (Placeholders) ---
    dep_check_source_label_data = DummyOperator(task_id="dep_check_source_label_data")
    bronze_label_store = DummyOperator(task_id="bronze_label_store")
    silver_label_store = DummyOperator(task_id="silver_label_store")
    gold_label_store = DummyOperator(task_id="gold_label_store")
    label_store_completed = DummyOperator(task_id="label_store_completed")

    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # --- Feature Store (Placeholders) ---
    dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_bronze_1")
    dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_bronze_2")
    dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_bronze_3")

    bronze_table_1 = DummyOperator(task_id="bronze_table_1")
    bronze_table_2 = DummyOperator(task_id="bronze_table_2")
    bronze_table_3 = DummyOperator(task_id="bronze_table_3")

    silver_table_1 = DummyOperator(task_id="silver_table_1")
    silver_table_2 = DummyOperator(task_id="silver_table_2")

    gold_feature_store = DummyOperator(task_id="gold_feature_store")
    feature_store_completed = DummyOperator(task_id="feature_store_completed")

    dep_check_source_data_bronze_1 >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_2 >> bronze_table_2 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_bronze_3 >> bronze_table_3 >> silver_table_2 >> gold_feature_store
    gold_feature_store >> feature_store_completed

    # --- Model Training ---
    model_training = BashOperator(
        task_id='run_model_training',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 train_model.py'
        ),
    )

    # --- Model Inference (Latest 3 OOT files) ---
    model_inference = BashOperator(
        task_id='run_model_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py --latest_oot 3 --modelname logistic_regression'
        ),
    )

    # --- Model Monitoring (Compare predictions vs ground truth) ---
    model_monitoring = BashOperator(
        task_id='run_model_monitoring',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 monitor.py'
        ),
    )

    # --- Generate Monitoring Visualization ---
    monitoring_visualization = BashOperator(
        task_id='run_monitoring_visualization',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 monitoring_visualization.py'
        ),
    )

    # --- Completion Flag ---
    pipeline_completed = DummyOperator(task_id="pipeline_completed")

    # --- DAG Flow ---
    # Data preparation
    feature_store_completed >> model_training
    label_store_completed >> model_training
    
    # ML Pipeline: Train -> Inference -> Monitor -> Visualize
    model_training >> model_inference >> model_monitoring >> monitoring_visualization >> pipeline_completed
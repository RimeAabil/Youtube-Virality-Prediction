from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.environ.get('AIRFLOW_HOME', '/opt/airflow'))

# Import V2 logic
from data.ingest_v2 import YouTubeIngestor
from ml.training.train import run_production_training

default_args = {
    'owner': 'ai_engineer',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'youtube_ai_virality_v2',
    default_args=default_args,
    description='Automated Production Pipeline for YouTube AI Virality',
    schedule_interval='@daily',
    catchup=False,
    tags=['production', 'ml', 'v2'],
) as dag:

    def ingest_v2_task():
        ingestor = YouTubeIngestor()
        count = ingestor.collect_niche_data()
        print(f"Successfully ingested {count} real videos.")

    def train_v2_task():
        run_production_training()
        print("Models successfully re-trained on new data.")

    t1 = PythonOperator(
        task_id='daily_scraping_real_data',
        python_callable=ingest_v2_task,
    )

    t2 = PythonOperator(
        task_id='model_retraining_production',
        python_callable=train_v2_task,
    )

    t1 >> t2

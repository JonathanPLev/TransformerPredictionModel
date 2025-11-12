"""
To run the dag in ../dag_setup
    - set AIRFLOW_HOME=~/CursorProjects/webscrape/airflow
    - airflow scheduler

For airflow UI
airflow apiserver --port 8080
"""
"""
To run the dag in ../dag_setup
    - set AIRFLOW_HOME=~/CursorProjects/webscrape/airflow
    - airflow standalone

For airflow UI
airflow apiserver --port 8080
"""

from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys, os

API_SCRIPTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../airflow_pipeline/api_scripts'))
if API_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, API_SCRIPTS_PATH)

from migrate_to_postgres import __main__ as migrate_to_postgres_main

default_args = {
    'owner': 'LineDancers',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='nba_sportsbook_pipeline',
    default_args=default_args,
    description='Daily NBA sportsbook data scraping and collection pipeline',
    schedule="0 22 * * *",  # <--- use cron string directly
    catchup=False,
    tags=['nba', 'betting', 'data-pipeline'],
)

def migrate_to_postgres_task():
    table_name = 'player_lines'
    migrate_to_postgres_main(table_name)

task_migrate_to_postgres = PythonOperator(
    task_id='migrate_to_postgres',
    python_callable=migrate_to_postgres_task,
    dag=dag,
)

# Only one task now, no fetch tasks needed
print("File ran?")

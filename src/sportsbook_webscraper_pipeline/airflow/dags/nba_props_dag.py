from airflow.models.dag import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import logging

API_SCRIPTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../airflow_pipeline/api_scripts")
)
if API_SCRIPTS_PATH not in sys.path:
    sys.path.insert(0, API_SCRIPTS_PATH)

# `# noqa: E402` tells the linter to ignore this specific rule here (need API_SCRIPTS_PATH before importing)
from migrate_to_postgres import __main__ as migrate_to_postgres_main # noqa: E402 

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
    schedule="0 22 * * *", 
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

check = logging.getLogger(__name__)

check.info("NBA props DAG initialized.")
check.debug("DAG file loaded successfully.")
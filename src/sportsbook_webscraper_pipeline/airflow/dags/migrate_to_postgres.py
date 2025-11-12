from sqlalchemy import create_engine, text as sql_text, Table, MetaData
import pandas as pd
import os
import sys
from dotenv import load_dotenv

# Ensure project root is importable when executed by Airflow or standalone
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow_pipeline.api_scripts.fetch_bettingpros import get_bettingpros_df
from airflow_pipeline.api_scripts.fetch_prizepicks import get_prizepicks_df
from airflow_pipeline.api_scripts.fetch_draftedge import get_draftedge_df

load_dotenv()
db_username = os.getenv("DB_USERNAME")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")
db_name = os.getenv("DB_NAME", "nba_deeplearning")

connection_string = (
    f'postgresql+psycopg2://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'
)

db_eng = create_engine(connection_string)
print("Successfully created db engine.")

def append_to_postgres(df, table_name):
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=db_eng)

    with db_eng.begin() as conn:  # transaction automatically commits
        for row in df.to_dict(orient='records'):
            conn.execute(table.insert(), row)
            
def check_if_table_exists(table_name):
    query = sql_text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name=:table)"
    )
    with db_eng.connect() as conn:
        result = conn.execute(query, {"table": table_name})
        return result.scalar()

def check_df_columns(df):
    table_columns = ['player_name', 'team', 'sportsbook', 'line_score', 'game_start', 'time_scraped', 'opponent_team', 'line_type']
    df_columns = df.columns.tolist()
    return all(col in df_columns for col in table_columns) and len(df_columns) == len(table_columns)

def __main__(table_name):
    if not check_if_table_exists(table_name):
        raise Exception(f"Table {table_name} does not exist.")

    dfs = [get_bettingpros_df(), get_prizepicks_df(), get_draftedge_df()]
    for df in dfs:
        if df.empty:
            print(f"Skipping empty DataFrame...")
            continue
        if not check_df_columns(df):
            raise Exception(f"df columns do not match WITH {table_name} attributes.")
        append_to_postgres(df, table_name)

        print(f"Successfully appended dataframe of length {len(df)} to {table_name} in postgres.")


if __name__ == "__main__":
    __main__('player_lines')

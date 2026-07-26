from sqlalchemy import create_engine, text as sql_text, Table, MetaData
from sqlalchemy.engine import URL
import os
import sys
from dotenv import load_dotenv
from airflow_pipeline.api_scripts.fetch_bettingpros import get_bettingpros_df
from airflow_pipeline.api_scripts.fetch_prizepicks import get_prizepicks_df
from airflow_pipeline.api_scripts.fetch_draftedge import get_draftedge_df

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv()

db_username = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("POSTGRES_EXTERNAL_PORT", "5432")
db_name = os.getenv("POSTGRES_DB")

url = URL.create(
    drivername="postgresql+psycopg2",
    username=db_username,
    password=db_password,
    host=db_host,
    port=db_port,
    database=db_name,
)

db_eng = create_engine(url)
print("Successfully created DB engine.")

def append_to_postgres(df, table_name):
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=db_eng)

    records = df.to_dict(orient='records')
    expected_count = len(records)

    with db_eng.begin() as conn: 
        result = conn.execute(table.insert(), records)
        if result.rowcount != -1:
            if result.rowcount != expected_count:
                raise Exception(
                    f"DB Verification Failed: Expected to write {expected_count} rows, "
                    f"but DB reported {result.rowcount} rows affected."
                )
        
        return expected_count
            
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
            print("Skipping empty DataFrame...")
            continue
        if not check_df_columns(df):
            raise Exception(f"df columns do not match WITH {table_name} attributes.")
        
        rows_inserted = append_to_postgres(df, table_name)
        
        if rows_inserted != len(df):
            raise Exception(f"Verification Mismatch: DataFrame had {len(df)} rows but function reported {rows_inserted}.")

        print(f"Successfully appended {rows_inserted} records to {table_name} in postgres.")



if __name__ == "__main__":
    __main__('player_lines')

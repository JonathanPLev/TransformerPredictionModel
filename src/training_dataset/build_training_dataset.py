import sqlalchemy as sqla

# uncomment to connect to the sportsbook database, make sure to ssh tunnel if connecting locally and not from server
# engine1 = sqla.create_engine("postgresql+psycopg2://line_dancer:sportsbook_data@localhost:5433/nba_deeplearning")

# to connect straight from server 
# engine2 = sqla.create_engine("postgresql+psycopg2://nba:nba@localhost:5432/nba")

# to connect from local machine - make sure to ssh tunnel to 55432 (or change connection string gitif you're using a different port)
engine2 = sqla.create_engine("postgresql+psycopg2://nba:nba@127.0.0.1:55432/nba")

def get_training_data():
    with engine2.connect() as conn:
        result = conn.execute(sqla.text("SELECT * FROM games;"))
        return result.fetchall()

print(len(get_training_data()))
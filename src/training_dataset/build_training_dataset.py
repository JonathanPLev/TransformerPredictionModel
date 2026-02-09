import sqlalchemy as sqla

# uncomment to connect to the sportsbook database, make sure to ssh tunnel if connecting locally and not from server
# engine1 = sqla.create_engine("postgresql+psycopg2://line_dancer:sportsbook_data@localhost:5433/nba_deeplearning")

# connect to the nba database - also make sure to ssh tunnel if connecting locally and not from server
engine2 = sqla.create_engine("postgresql+psycopg2://nba:nba@localhost:5432/nba")
    

def get_training_data():
    with engine2.connect() as conn:
        result = conn.execute(sqla.text("SELECT * FROM games;"))
        return result.fetchall()

print(len(get_training_data()))
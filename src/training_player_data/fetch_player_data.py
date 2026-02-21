from nba_api.stats.static import players
import pandas as pd
import sqlalchemy as sqla

engine = sqla.create_engine("postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba")

def fetch_player_data(player_name: str):
    player_found = players.find_players_by_full_name(player_name)

    if not player_found:
        raise ValueError(f"Player {player_name} not found")
    
    elif (len(player_found) > 1):
        print("Founds multipler players, try again with a specific name:")
        print(player_found)
        exit()

    player_id = player_found[0]["id"]

    with engine.connect() as conn:
        result = conn.execute(
            sqla.text("""
                SELECT *
                FROM player_statistics
                WHERE person_id = :pid
                ORDER BY game_date ASC
            """),
            {"pid": player_id}
        )

        cols = result.keys()
        return pd.DataFrame(result.fetchall(), columns=cols)

print(fetch_player_data("LeBron James"))
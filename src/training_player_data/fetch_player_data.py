from nba_api.stats.static import players
import pandas as pd
import sqlalchemy as sqla
from datetime import datetime

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
        player_data = conn.execute(
            sqla.text("""
                        SELECT person_id, game_date, points, num_minutes, assists, rebounds_total, turnovers,
                               field_goals_attempted, three_pointers_attempted, free_throws_attempted, steals,
                               blocks, plus_minus_points
                        FROM player_statistics
                        WHERE person_id = :pid
                        ORDER BY game_date ASC;
            """),
            {"pid": player_id}
        )

        cols = player_data.keys()

        features = pd.DataFrame(player_data.fetchall(), columns=cols)

        player_team = conn.execute(
            sqla.text("""
                        SELECT person_id, player_team_name, game_id, game_date
                        FROM player_statistics
                        WHERE person_id = :pid
                        ORDER BY game_date DESC
                        LIMIT 1;
            """),
            {"pid": player_id}
        )

        # is player currently playing?
        team_name = player_team.fetchone()[1]
        
        # return team_name

        current_time = datetime.now() 
        
        # we are grabbing the games that happen after current time
        game_date = conn.execute(
            sqla.text("""
                        SELECT gameid, NULLIF(gamedatetimeest,'')::timestamp AS game_ts_est, hometeamname, awayteamname
                        FROM games_raw
                        WHERE (:player_team = hometeamname OR :player_team = awayteamname)
                        AND NULLIF(gamedatetimeest,'')::timestamp > :time
                        ORDER BY game_ts_est DESC;
            """),
            {"player_team": team_name, "time": current_time}
        )


        return game_date.fetchall()

        # also pull the next opponent game

print(fetch_player_data("LeBron James"))
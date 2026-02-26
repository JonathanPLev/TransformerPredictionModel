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


        player_games = pd.DataFrame(game_date.fetchall(), columns=game_date.keys())
        player_games = player_games.sort_values("game_ts_est")  
        
        future_games = player_games[player_games["game_ts_est"] > current_time].copy()

        started_games = player_games[player_games["game_ts_est"] <= current_time].copy()

        # find "today game" = latest game that started recently (within ~3h)
        IN_PROGRESS_HOURS = 3
        in_progress = started_games[
            started_games["game_ts_est"] >= (current_time - pd.Timedelta(hours=IN_PROGRESS_HOURS))
        ]

        if len(in_progress) > 0:
            # currently playing -> target is next game AFTER the in-progress game
            today_game = in_progress.sort_values("game_ts_est").iloc[-1]
            future_after_today = player_games[player_games["game_ts_est"] > today_game["game_ts_est"]]
            if future_after_today.empty:
                return None
            target_game = future_after_today.sort_values("game_ts_est").iloc[0]
            rest_anchor_date = today_game["game_ts_est"].date()
        else:
            # not currently playing -> target is next future game
            if future_games.empty:
                return None
            target_game = future_games.sort_values("game_ts_est").iloc[0]
            rest_anchor_date = features["game_date"].iloc[-1]  

        

print(fetch_player_data("LeBron James"))
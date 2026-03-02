from nba_api.stats.static import players
import pandas as pd
import sqlalchemy as sqla
from datetime import datetime
import nbainjuries
from zoneinfo import ZoneInfo

engine = sqla.create_engine("postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba")

"""
nbainjuries requires JVM, make sure to export path for the library to return report

eg: ❯ export JAVA_HOME=$(/usr/libexec/java_home -v 18 -a arm64)                                                                              ─╯
    ❯ echo "$JAVA_HOME"                                                                                                                      ─╯
"""
def injury_status(player_name_first: str, player_name_last: str) -> pd.DataFrame | None:
    eastern_time_zone = ZoneInfo("America/New_York")
    current_time = datetime.now(eastern_time_zone)

    minute = 30 if current_time.minute >= 30 else 0
    timestamp = current_time.replace(minute=minute, second=0, microsecond=0)

    report = nbainjuries.injury.get_reportdata(timestamp=timestamp.replace(tzinfo=None), return_df=True)

    player_name = f"{player_name_last.strip()}, {player_name_first.strip()}".lower()
    names = report["Player Name"].astype(str).str.strip().str.lower()
    
    player_report = report[names == player_name]

    if player_report.empty:
        return None
    else:
        return player_report
    
# print(injury_status("Stephen", "Curry"))
# print(injury_status("Lebron", "James"))

def fetch_player_data(player_name: str) -> tuple[pd.DataFrame, str]:

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
        
        team_name = player_team.fetchone()[1]

        return (features, team_name)
        

# features, team = fetch_player_data("LeBron James")
# print(features.tail())

def schedule_status(features, team_name):
    current_time = datetime.now()

    features = features.copy()
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce")
    features = features.dropna(subset=["game_date"]).sort_values("game_date")

    if features.empty:
        return None

    
    start_time = current_time - pd.Timedelta(days=2)
    end_time = current_time + pd.Timedelta(days=14)

    with engine.connect() as conn:
        game_rows = conn.execute(
            sqla.text("""
                SELECT
                    gameid,
                    NULLIF(gamedatetimeest,'')::timestamp AS game_ts_est,
                    hometeamname,
                    awayteamname
                FROM games_raw
                WHERE (:player_team = hometeamname OR :player_team = awayteamname)
                  AND NULLIF(gamedatetimeest,'')::timestamp BETWEEN :start_time AND :end_time
                ORDER BY game_ts_est ASC;
            """),
            {"player_team": team_name, "start_time": start_time, "end_time": end_time},
        ).fetchall()

    if not game_rows:
        return None

    player_games = pd.DataFrame(game_rows, columns=["gameid", "game_ts_est", "hometeamname", "awayteamname"])
    player_games = player_games.dropna(subset=["game_ts_est"]).sort_values("game_ts_est")

    # Split games
    future_games = player_games[player_games["game_ts_est"] > current_time].copy()
    started_games = player_games[player_games["game_ts_est"] <= current_time].copy()

    # Detect in-progress game: started within last X hours
    IN_PROGRESS_HOURS = 3
    in_progress = started_games[started_games["game_ts_est"] >= (current_time - pd.Timedelta(hours=IN_PROGRESS_HOURS))]

    currently_playing = not in_progress.empty

    if currently_playing:
        # If currently playing, anchor rest at the START time of the current game
        current_game = in_progress.iloc[-1]
        anchor_dt = current_game["game_ts_est"]

        future_after = player_games[player_games["game_ts_est"] > anchor_dt]
        if future_after.empty:
            return None
        target_game = future_after.iloc[0]
    else:
    
        anchor_dt = features["game_date"].iloc[-1]

        if future_games.empty:
            return None
        target_game = future_games.iloc[0]

    # Compute days_rest as difference in calendar days between anchor date and next game date
    next_dt = pd.to_datetime(target_game["game_ts_est"])
    anchor_date = pd.to_datetime(anchor_dt).date()
    next_date = next_dt.date()
    days_rest = max(0, (next_date - anchor_date).days)

    # Opponent + home/away
    home_team = target_game["hometeamname"]
    away_team = target_game["awayteamname"]
    is_home = (team_name == home_team)
    opponent = away_team if is_home else home_team

    return {
        "currently_playing": currently_playing,
        "days_rest": int(days_rest),
        "next_game_id": int(target_game["gameid"]),
        "next_game_ts_est": str(target_game["game_ts_est"]),
        "opponent": opponent,
        "is_home": bool(is_home),
        "home_team": home_team,
        "away_team": away_team,
    }

def build_prediction_inputs(player_name: str):
    try:
        # 1) stats + team
        features, team_name = fetch_player_data(player_name)
        if features is None or features.empty or not team_name:
            return None, 404

        # 2) official name -> first/last
        found = players.find_players_by_full_name(player_name)
        if not found or len(found) != 1:
            return None, 404

        full_name = found[0].get("full_name") or player_name
        parts = full_name.split()
        if len(parts) < 2:
            return None, 404
        first_name, last_name = parts[0], parts[-1]

        # 3) injury
        inj = injury_status(first_name, last_name)
        if inj is not None:
            print(inj[["Player Name", "Current Status", "Reason"]].head(1))
            return None, 422

        # 4) schedule (days_rest + opponent)
        sched = schedule_status(features, team_name)
        if sched is None:
            return None, 404

        out = features.copy()
        out["days_rest"] = sched["days_rest"]
        out["opponent"] = sched["opponent"]
        return out, 200

    except Exception:
        return None, 404

df, code = build_prediction_inputs("Stephen Curry")
print(code)
if df is not None:
    print(df[["person_id","game_date","points","days_rest","opponent"]].tail())
from nba_api.stats.static import players
import pandas as pd
import sqlalchemy as sqla
from datetime import datetime
import nbainjuries
from zoneinfo import ZoneInfo
from pathlib import Path

engine = sqla.create_engine("postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCHEDULE_CSV = ROOT / "nba_dataset" / "LeagueSchedule25_26.csv"

_SCHEDULE_DF: pd.DataFrame | None = None
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
                        WHERE person_id = :pid AND game_type = 'Regular Season'
                        AND game_date IS NOT NULL
                        AND player_team_name IS NOT NULL
                        ORDER BY game_date DESC
                        LIMIT 1;
            """),
            {"pid": player_id}
        )
        
        team_name = player_team.fetchone()[1]

        return (features, team_name)
        

# features, team = fetch_player_data("LeBron James")
# print(features.tail())

def _load_schedule_df() -> pd.DataFrame:
    global _SCHEDULE_DF
    if _SCHEDULE_DF is not None:
        return _SCHEDULE_DF

    df = pd.read_csv(SCHEDULE_CSV)

    # Normalize types / names
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
    df["homeTeamName"] = df["homeTeamName"].astype(str).str.strip()
    df["awayTeamName"] = df["awayTeamName"].astype(str).str.strip()

    # IDs sometimes come in as floats from CSV; force numeric -> Int64
    for c in ["gameId", "homeTeamId", "awayTeamId"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Drop rows without datetime
    df = df.dropna(subset=["gameDateTimeEst"])

    _SCHEDULE_DF = df
    return df


def _infer_team_id_from_csv(schedule_df: pd.DataFrame, team_name: str) -> int | None:
    """
    Map team_name -> NBA teamId using schedule rows.
    """
    home_ids = schedule_df.loc[schedule_df["homeTeamName"] == team_name, "homeTeamId"].dropna()
    away_ids = schedule_df.loc[schedule_df["awayTeamName"] == team_name, "awayTeamId"].dropna()
    ids = pd.concat([home_ids, away_ids], ignore_index=True)

    if ids.empty:
        return None

    # Most common id wins
    return int(ids.value_counts().idxmax())


def schedule_status(features: pd.DataFrame, team_name: str):
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)

    # Prep features
    features = features.copy()
    features["game_date"] = pd.to_datetime(features["game_date"], errors="coerce")
    features = features.dropna(subset=["game_date"]).sort_values("game_date")
    if features.empty:
        return None

    schedule_df = _load_schedule_df()
    team_id = _infer_team_id_from_csv(schedule_df, team_name)
    if team_id is None:
        print(f"schedule_status(csv): cannot find team_id for team_name={team_name!r} in {SCHEDULE_CSV}")
        return None

    # Time window
    start_time = (now - pd.Timedelta(days=2)).replace(tzinfo=None)   # schedule CSV is naive EST
    end_time   = (now + pd.Timedelta(days=14)).replace(tzinfo=None)

    # Filter schedule
    team_games = schedule_df[
        ((schedule_df["homeTeamId"] == team_id) | (schedule_df["awayTeamId"] == team_id)) &
        (schedule_df["gameDateTimeEst"].between(start_time, end_time))
    ].copy()

    print("schedule_status(csv):", team_name, "team_id=", team_id, "rows=", len(team_games),
          "window=", start_time, "->", end_time)

    if team_games.empty:
        return None

    team_games = team_games.sort_values("gameDateTimeEst")

    # For comparisons, localize naive EST to NY tz
    team_games["game_ts_est"] = pd.to_datetime(team_games["gameDateTimeEst"]).dt.tz_localize(tz)

    future_games = team_games[team_games["game_ts_est"] > now].copy()
    started_games = team_games[team_games["game_ts_est"] <= now].copy()

    # in-progress: started within last X hours
    IN_PROGRESS_HOURS = 3
    in_progress = started_games[started_games["game_ts_est"] >= (now - pd.Timedelta(hours=IN_PROGRESS_HOURS))]
    currently_playing = not in_progress.empty

    if currently_playing:
        anchor_dt = in_progress.iloc[-1]["game_ts_est"]
        future_after = team_games[team_games["game_ts_est"] > anchor_dt]
        if future_after.empty:
            return None
        target = future_after.iloc[0]
    else:
        # anchor at last played game_date from stats
        anchor_dt = pd.to_datetime(features["game_date"].iloc[-1]).tz_localize(tz)
        if future_games.empty:
            return None
        target = future_games.iloc[0]

    # days_rest in calendar days
    days_rest = max(0, (target["game_ts_est"].date() - anchor_dt.date()).days)

    home_team = str(target["homeTeamName"])
    away_team = str(target["awayTeamName"])
    is_home = (team_name == home_team)
    opponent = away_team if is_home else home_team
    opponent_id = int(target["awayTeamId"] if is_home else target["homeTeamId"])

    return {
        "currently_playing": currently_playing,
        "days_rest": int(days_rest),
        "next_game_id": int(target["gameId"]),
        "next_game_ts_est": str(target["game_ts_est"]),
        "opponent": opponent,
        "opponent_id": opponent_id,
        "is_home": bool(is_home),
        "home_team": home_team,
        "away_team": away_team,
    }

def build_prediction_inputs(player_name: str):
    try:
        features, team_name = fetch_player_data(player_name)
        print("    team_name =", team_name)
        print("    features rows =", 0 if features is None else len(features))
        if features is None or features.empty:
            return None, None, None, 404

        # injury check
        found = players.find_players_by_full_name(player_name)
        print("    found =", found)
        if not found:
            print("    ERROR: nba_api did not find player")
            return None, None, None, 404

        full_name = found[0]["full_name"]
        first_name, last_name = full_name.split()[0], full_name.split()[-1]

        inj = injury_status(first_name, last_name)
        if inj is not None:
            status = str(inj.iloc[0].get("Current Status", "")).strip().lower()
            if status in {"out", "out for season", "inactive"}:
                return None, None, None, 422

        sched = schedule_status(features, team_name)
        if sched is None:
            # still return player stats, but no schedule features
            return features, None, None, 200

        return features, sched["days_rest"], sched["opponent_id"], 200

    except Exception:
        return None, None, None, 404

 


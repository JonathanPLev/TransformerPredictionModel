from pathlib import Path
import pandas as pd
import numpy as np

def fetch_player_df(query: str):
    """
    TEMP STUB:
      - if query is digits -> load that player from PlayerStatistics.csv
      - else -> 404 (not built yet)
    returns: (df, meta, status_code)
    """
    s = str(query).strip()

    if not s.isdigit():
        return None, {"reason": "name lookup not wired yet; send numeric personId"}, 404

    person_id = int(s)

    root = Path(__file__).resolve().parents[2]  # repo root
    data_dir = root / "src" / "nba_dataset" / "Data"
    stats_path = data_dir / "PlayerStatistics.csv"
    games_path = data_dir / "Games.csv"

    if not stats_path.exists() or not games_path.exists():
        return None, {"reason": "missing nba_dataset/Data CSVs"}, 404

    df_stats = pd.read_csv(stats_path, low_memory=False)
    df_games = pd.read_csv(games_path, low_memory=False)

    df_player = df_stats[df_stats["personId"] == person_id].copy()
    if df_player.empty:
        return None, {"reason": f"no rows for personId={person_id}"}, 404

    # add opp_id like training expects
    g = df_games[["gameId", "hometeamId", "awayteamId"]].copy()
    df_player = df_player.merge(g, on="gameId", how="left")
    df_player["home"] = pd.to_numeric(df_player["home"], errors="coerce").fillna(0).astype(int)

    df_player["opp_id"] = np.where(
        df_player["home"] == 1,
        df_player["awayteamId"],
        df_player["hometeamId"],
    )
    df_player["opp_id"] = df_player["opp_id"].fillna(-1).astype(int).astype(str)

    return df_player, {"person_id": person_id}, 200
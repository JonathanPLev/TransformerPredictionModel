"""
NBA Player Transformer — Data Pipeline

Queries the NBA PostgreSQL database and produces train.pt / val.pt tensors
ready for use with model.py.

Each sample is a K=10 game sliding window with 53 continuous features per
timestep plus categorical IDs, targeting next-game [points, assists,
rebounds, minutes].

Usage:
    python data_pipeline.py
    python data_pipeline.py --max-players 50 --val-frac 0.2

Outputs:
    train.pt, val.pt       – torch tensor dicts
    pipeline_meta.pkl      – scaler + encoders needed at inference time
"""

import argparse
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE_URL = "postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba"
SEQ_LEN = 10
MIN_GAMES_PER_PLAYER = 25
VAL_FRAC = 0.20
TEST_FRAC = 0.05
ROOKIE_FILL_SENTINEL = -1  # replaced with per-stat mean later

CONT_FEATURE_NAMES = [
    # Lag values – last 5 games (20)
    "points_lag_1", "points_lag_2", "points_lag_3", "points_lag_4", "points_lag_5",
    "rebounds_lag_1", "rebounds_lag_2", "rebounds_lag_3", "rebounds_lag_4", "rebounds_lag_5",
    "assists_lag_1", "assists_lag_2", "assists_lag_3", "assists_lag_4", "assists_lag_5",
    "minutes_lag_1", "minutes_lag_2", "minutes_lag_3", "minutes_lag_4", "minutes_lag_5",
    # Current-season averages (4)
    "points_avg_current_season",
    "rebounds_avg_current_season",
    "assists_avg_current_season",
    "minutes_avg_current_season",
    # Previous-season averages (4)
    "points_avg_previous_season",
    "rebounds_avg_previous_season",
    "assists_avg_previous_season",
    "minutes_avg_previous_season",
    # Rolling last-5 averages (4)
    "points_avg_last_5",
    "rebounds_avg_last_5",
    "assists_avg_last_5",
    "minutes_avg_last_5",
    # Trends over last 5 games (linear slope) (4)
    "points_trend_5",
    "rebounds_trend_5",
    "assists_trend_5",
    "minutes_trend_5",
    # Std over last 5 games (4)
    "points_std_last_5",
    "rebounds_std_last_5",
    "assists_std_last_5",
    "minutes_std_last_5",
    # Recent performance vs season average (4)
    "points_vs_season_avg",
    "rebounds_vs_season_avg",
    "assists_vs_season_avg",
    "minutes_vs_season_avg",
    # Game context (2)
    "is_home_game",
    "days_since_last_game",
    # Efficiency averages – current season (4)
    "fg_pct_avg_current_season",
    "three_pct_avg_current_season",
    "ft_pct_avg_current_season",
    "plus_minus_avg_current_season",
    # Team context (3)
    "team_points_allowed_current_season",
    "opponent_points_allowed_current_season",
    "team_pace_current_season",
]

NUM_CONT = len(CONT_FEATURE_NAMES)  # 53


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_season(date: pd.Timestamp) -> int:
    """NBA season year: Oct–Jun → year the season started."""
    return date.year if date.month >= 10 else date.year - 1


def linear_slope(values: np.ndarray) -> float:
    """Least-squares slope of values over index 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    denom = (x * x).sum()
    return float((x * values).sum() / denom) if denom != 0 else 0.0


def safe_pct(made: float, attempted: float) -> float:
    return made / attempted if attempted > 0 else 0.0


# ---------------------------------------------------------------------------
# Step 1: Load all player game data
# ---------------------------------------------------------------------------

def load_player_games(engine, min_games: int, max_players: Optional[int]) -> pd.DataFrame:
    """Return all qualifying player game rows ordered by (person_id, game_date)."""
    limit_clause = f"LIMIT {max_players}" if max_players else ""
    eligible_sql = text(f"""
        SELECT person_id
        FROM player_statistics
        WHERE points IS NOT NULL
          AND num_minutes > 0
          AND game_date IS NOT NULL
        GROUP BY person_id
        HAVING COUNT(*) >= :min_games
        ORDER BY person_id
        {limit_clause}
    """)
    with engine.connect() as conn:
        eligible = pd.DataFrame(conn.execute(eligible_sql, {"min_games": min_games}).fetchall(),
                                columns=["person_id"])

    if eligible.empty:
        raise RuntimeError("No eligible players found.")

    pids = tuple(eligible["person_id"].tolist())
    print(f"  Found {len(pids)} eligible players.")

    games_sql = text("""
        SELECT
            ps.person_id,
            ps.game_id,
            ps.game_date,
            ps.points,
            ps.rebounds_total      AS rebounds,
            ps.assists,
            ps.num_minutes         AS minutes,
            ps.field_goals_made,
            ps.field_goals_attempted,
            ps.three_pointers_made,
            ps.three_pointers_attempted,
            ps.free_throws_made,
            ps.free_throws_attempted,
            ps.plus_minus_points   AS plus_minus,
            ps.home                AS is_home,
            ps.player_team_name,
            ps.opponent_team_name
        FROM player_statistics ps
        WHERE ps.person_id = ANY(:pids)
          AND ps.points IS NOT NULL
          AND ps.num_minutes > 0
          AND ps.game_date IS NOT NULL
        ORDER BY ps.person_id, ps.game_date
    """)
    with engine.connect() as conn:
        df = pd.DataFrame(
            conn.execute(games_sql, {"pids": list(pids)}).fetchall(),
            columns=[
                "person_id", "game_id", "game_date",
                "points", "rebounds", "assists", "minutes",
                "fg_made", "fg_attempted", "three_made", "three_attempted",
                "ft_made", "ft_attempted", "plus_minus",
                "is_home", "player_team_name", "opponent_team_name",
            ],
        )

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["is_home"] = df["is_home"].astype(int)
    df["season"] = df["game_date"].apply(get_season)

    # Fill nulls in counting stats with 0
    for col in ["points", "rebounds", "assists", "minutes",
                "fg_made", "fg_attempted", "three_made", "three_attempted",
                "ft_made", "ft_attempted", "plus_minus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    print(f"  Loaded {len(df):,} game rows.")
    return df


# ---------------------------------------------------------------------------
# Step 2: Precompute team defensive stats & pace
# ---------------------------------------------------------------------------

def compute_team_game_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player rows to team-game totals.
    Returns one row per (team, game_id, game_date, season).
    """
    agg = (
        df.groupby(["player_team_name", "game_id", "game_date", "season"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_fg_attempted=("fg_attempted", "sum"),
            team_ft_attempted=("ft_attempted", "sum"),
            team_turnovers=("plus_minus", "count"),  # row count ≈ roster size; unused
        )
    )
    # Pace proxy: FGA + 0.4*FTA + turnovers (we use fg_attempted + 0.4*ft_attempted)
    # We don't have true turnovers at team level easily, so simplify
    agg["pace_proxy"] = agg["team_fg_attempted"] + 0.4 * agg["team_ft_attempted"]
    return agg


def build_team_defensive_lookup(df: pd.DataFrame, team_totals: pd.DataFrame):
    """
    For each (team, game_date, season), compute rolling current-season averages
    of points_allowed and pace UP TO (not including) that date.

    points_allowed by Team X = avg total points scored by opponents against Team X.
    The opponent's scoring = rows where opponent_team_name == X, summed by game.

    Returns a dict: (team_name, game_date, season) → {pts_allowed, pace}
    """
    # Points allowed by team X: games where opponent_team_name = X, summed by game
    opp_scoring = (
        df.groupby(["opponent_team_name", "game_id", "game_date", "season"], as_index=False)
        .agg(points_scored_against=("points", "sum"))
        .rename(columns={"opponent_team_name": "team_name"})
    )

    # Merge pace from team_totals
    pace_df = team_totals[["player_team_name", "game_id", "game_date", "season", "pace_proxy"]].copy()
    pace_df = pace_df.rename(columns={"player_team_name": "team_name"})

    # Get all (team, game_date, season) combinations we need to answer
    # For each team game, compute cumulative season avg up to (not including) that date
    all_teams = sorted(df["player_team_name"].dropna().unique())
    all_seasons = sorted(df["season"].unique())

    lookup: dict = {}

    for team in all_teams:
        team_opp = opp_scoring[opp_scoring["team_name"] == team].sort_values("game_date")
        team_pace = pace_df[pace_df["team_name"] == team].sort_values("game_date")

        for season in all_seasons:
            s_opp = team_opp[team_opp["season"] == season].reset_index(drop=True)
            s_pace = team_pace[team_pace["season"] == season].reset_index(drop=True)

            # Get all dates this team has games in this season
            team_season_games = df[
                (df["player_team_name"] == team) & (df["season"] == season)
            ]["game_date"].drop_duplicates().sort_values()

            for gdate in team_season_games:
                prior_opp = s_opp[s_opp["game_date"] < gdate]
                prior_pace = s_pace[s_pace["game_date"] < gdate]

                pts_allowed = (
                    prior_opp["points_scored_against"].mean()
                    if not prior_opp.empty else 110.0
                )
                pace = (
                    prior_pace["pace_proxy"].mean()
                    if not prior_pace.empty else 85.0
                )
                lookup[(team, gdate, season)] = {
                    "team_points_allowed": float(pts_allowed),
                    "team_pace": float(pace),
                }

    return lookup


# ---------------------------------------------------------------------------
# Step 3: Previous season averages (per player per season)
# ---------------------------------------------------------------------------

def compute_season_averages(df: pd.DataFrame) -> dict:
    """
    Returns {(person_id, season): {stat: avg, ...}} for each player-season.
    Used to fill 'previous season' features.
    """
    avgs = {}
    for (pid, season), gdf in df.groupby(["person_id", "season"]):
        avgs[(pid, season)] = {
            "points": gdf["points"].mean(),
            "rebounds": gdf["rebounds"].mean(),
            "assists": gdf["assists"].mean(),
            "minutes": gdf["minutes"].mean(),
            "fg_pct": (gdf["fg_made"].sum() / gdf["fg_attempted"].sum()
                       if gdf["fg_attempted"].sum() > 0 else 0.0),
            "three_pct": (gdf["three_made"].sum() / gdf["three_attempted"].sum()
                          if gdf["three_attempted"].sum() > 0 else 0.0),
            "ft_pct": (gdf["ft_made"].sum() / gdf["ft_attempted"].sum()
                       if gdf["ft_attempted"].sum() > 0 else 0.0),
            "plus_minus": gdf["plus_minus"].mean(),
        }
    return avgs


def compute_global_stat_means(season_avgs: dict) -> dict:
    """Compute grand mean per stat across all player-seasons (for rookie fill)."""
    keys = ["points", "rebounds", "assists", "minutes",
            "fg_pct", "three_pct", "ft_pct", "plus_minus"]
    buckets = {k: [] for k in keys}
    for v in season_avgs.values():
        for k in keys:
            buckets[k].append(v[k])
    return {k: float(np.mean(vals)) if vals else 0.0 for k, vals in buckets.items()}


# ---------------------------------------------------------------------------
# Step 4: Build per-player feature rows
# ---------------------------------------------------------------------------

def build_player_feature_rows(
    player_df: pd.DataFrame,
    season_avgs: dict,
    global_means: dict,
    team_def_lookup: dict,
) -> list[dict]:
    """
    For a single player's game history (sorted by date), build one feature dict
    per game. Each dict uses only data from games BEFORE the current game.
    Returns list of row dicts, each containing all 53 continuous features plus
    metadata (person_id, game_date, season, game_idx, player_team_name,
    opponent_team_name, target_points, target_rebounds, target_assists,
    target_minutes).
    """
    rows = []
    stats = ["points", "rebounds", "assists", "minutes"]

    # Group by season to track season-cumulative stats
    for season, season_df in player_df.groupby("season"):
        season_df = season_df.sort_values("game_date").reset_index(drop=True)
        person_id = int(season_df["person_id"].iloc[0])

        prev_season = season - 1
        prev_avgs = season_avgs.get(
            (person_id, prev_season),
            {k: global_means[k] for k in global_means}  # rookie fill
        )

        # Running season accumulators (exclude current game)
        cum_pts, cum_reb, cum_ast, cum_min = [], [], [], []
        cum_fg_made, cum_fg_att = [], []
        cum_3_made, cum_3_att = [], []
        cum_ft_made, cum_ft_att = [], []
        cum_pm = []

        for i, row in season_df.iterrows():
            gdate = row["game_date"]

            # ---- Days since last game ----
            if len(cum_pts) == 0:
                days_rest = 7.0  # unknown first game
            else:
                prev_date = season_df.iloc[len(cum_pts) - 1]["game_date"]
                days_rest = float((gdate - prev_date).days)

            # ---- Season game index (1-based, capped at 82) ----
            game_idx = min(len(cum_pts) + 1, 82)

            # ---- Lag values (last 5 games) ----
            hist_pts  = cum_pts[-5:]
            hist_reb  = cum_reb[-5:]
            hist_ast  = cum_ast[-5:]
            hist_min  = cum_min[-5:]

            def lag(hist, k):
                """k-th lag (1=most recent). Returns 0 if not available."""
                idx = len(hist) - k
                return float(hist[idx]) if idx >= 0 else 0.0

            # ---- Current-season averages (up to but not including this game) ----
            def season_avg(hist):
                return float(np.mean(hist)) if hist else 0.0

            # ---- Rolling last-5 averages, trends, std ----
            def rolling_avg(hist):
                return float(np.mean(hist)) if hist else 0.0

            def rolling_trend(hist):
                return linear_slope(np.array(hist, dtype=float)) if len(hist) >= 2 else 0.0

            def rolling_std(hist):
                return float(np.std(hist)) if len(hist) >= 2 else 0.0

            # ---- Efficiency averages ----
            fg_pct_avg = safe_pct(sum(cum_fg_made), sum(cum_fg_att))
            three_pct_avg = safe_pct(sum(cum_3_made), sum(cum_3_att))
            ft_pct_avg = safe_pct(sum(cum_ft_made), sum(cum_ft_att))
            pm_avg = float(np.mean(cum_pm)) if cum_pm else 0.0

            # ---- Team defensive context ----
            player_team = row["player_team_name"]
            opp_team = row["opponent_team_name"]

            def_self = team_def_lookup.get((player_team, gdate, season), {})
            def_opp = team_def_lookup.get((opp_team, gdate, season), {})

            team_pts_allowed = def_self.get("team_points_allowed", 110.0)
            opp_pts_allowed = def_opp.get("team_points_allowed", 110.0)
            team_pace = def_self.get("team_pace", 85.0)

            # ---- Assemble continuous feature row ----
            s_pts  = season_avg(cum_pts)
            s_reb  = season_avg(cum_reb)
            s_ast  = season_avg(cum_ast)
            s_min  = season_avg(cum_min)

            l5_pts = rolling_avg(hist_pts)
            l5_reb = rolling_avg(hist_reb)
            l5_ast = rolling_avg(hist_ast)
            l5_min = rolling_avg(hist_min)

            feat = {
                # Lag values
                "points_lag_1": lag(cum_pts, 1),
                "points_lag_2": lag(cum_pts, 2),
                "points_lag_3": lag(cum_pts, 3),
                "points_lag_4": lag(cum_pts, 4),
                "points_lag_5": lag(cum_pts, 5),
                "rebounds_lag_1": lag(cum_reb, 1),
                "rebounds_lag_2": lag(cum_reb, 2),
                "rebounds_lag_3": lag(cum_reb, 3),
                "rebounds_lag_4": lag(cum_reb, 4),
                "rebounds_lag_5": lag(cum_reb, 5),
                "assists_lag_1": lag(cum_ast, 1),
                "assists_lag_2": lag(cum_ast, 2),
                "assists_lag_3": lag(cum_ast, 3),
                "assists_lag_4": lag(cum_ast, 4),
                "assists_lag_5": lag(cum_ast, 5),
                "minutes_lag_1": lag(cum_min, 1),
                "minutes_lag_2": lag(cum_min, 2),
                "minutes_lag_3": lag(cum_min, 3),
                "minutes_lag_4": lag(cum_min, 4),
                "minutes_lag_5": lag(cum_min, 5),
                # Current season averages
                "points_avg_current_season": s_pts,
                "rebounds_avg_current_season": s_reb,
                "assists_avg_current_season": s_ast,
                "minutes_avg_current_season": s_min,
                # Previous season averages
                "points_avg_previous_season": prev_avgs["points"],
                "rebounds_avg_previous_season": prev_avgs["rebounds"],
                "assists_avg_previous_season": prev_avgs["assists"],
                "minutes_avg_previous_season": prev_avgs["minutes"],
                # Rolling last-5
                "points_avg_last_5": l5_pts,
                "rebounds_avg_last_5": l5_reb,
                "assists_avg_last_5": l5_ast,
                "minutes_avg_last_5": l5_min,
                # Trends
                "points_trend_5": rolling_trend(hist_pts),
                "rebounds_trend_5": rolling_trend(hist_reb),
                "assists_trend_5": rolling_trend(hist_ast),
                "minutes_trend_5": rolling_trend(hist_min),
                # Std
                "points_std_last_5": rolling_std(hist_pts),
                "rebounds_std_last_5": rolling_std(hist_reb),
                "assists_std_last_5": rolling_std(hist_ast),
                "minutes_std_last_5": rolling_std(hist_min),
                # vs season avg
                "points_vs_season_avg": l5_pts - s_pts,
                "rebounds_vs_season_avg": l5_reb - s_reb,
                "assists_vs_season_avg": l5_ast - s_ast,
                "minutes_vs_season_avg": l5_min - s_min,
                # Game context
                "is_home_game": float(row["is_home"]),
                "days_since_last_game": days_rest,
                # Efficiency
                "fg_pct_avg_current_season": fg_pct_avg,
                "three_pct_avg_current_season": three_pct_avg,
                "ft_pct_avg_current_season": ft_pct_avg,
                "plus_minus_avg_current_season": pm_avg,
                # Team context
                "team_points_allowed_current_season": team_pts_allowed,
                "opponent_points_allowed_current_season": opp_pts_allowed,
                "team_pace_current_season": team_pace,
            }

            feat["person_id"] = person_id
            feat["game_date"] = gdate
            feat["season"] = season
            feat["game_idx"] = game_idx
            feat["player_team_name"] = player_team
            feat["opponent_team_name"] = opp_team
            # Targets
            feat["target_points"] = float(row["points"])
            feat["target_rebounds"] = float(row["rebounds"])
            feat["target_assists"] = float(row["assists"])
            feat["target_minutes"] = float(row["minutes"])

            rows.append(feat)

            # Update running accumulators (include current game for next iteration)
            cum_pts.append(row["points"])
            cum_reb.append(row["rebounds"])
            cum_ast.append(row["assists"])
            cum_min.append(row["minutes"])
            cum_fg_made.append(row["fg_made"])
            cum_fg_att.append(row["fg_attempted"])
            cum_3_made.append(row["three_made"])
            cum_3_att.append(row["three_attempted"])
            cum_ft_made.append(row["ft_made"])
            cum_ft_att.append(row["ft_attempted"])
            cum_pm.append(row["plus_minus"])

    return rows


# ---------------------------------------------------------------------------
# Step 5: Build K=10 sliding window sequences
# ---------------------------------------------------------------------------

def build_sequences(all_rows: list[dict], seq_len: int = SEQ_LEN) -> list[dict]:
    """
    Group rows by (person_id, season) and create sliding windows of length
    seq_len. The window [i-seq_len : i] predicts the game at index i.

    The target game's feature row is appended as a (K+1)th context token.
    Its continuous features (rolling stats, season avgs) are all computed from
    games before T (no leakage), and its game context features (home/away,
    opponent identity, opponent defensive stats, days rest) are known before
    the game happens.
    """
    from itertools import groupby

    # Sort chronologically
    all_rows.sort(key=lambda r: (r["person_id"], r["season"], r["game_date"]))

    sequences = []
    key_fn = lambda r: (r["person_id"], r["season"])

    for (pid, season), grp in groupby(all_rows, key=key_fn):
        season_rows = list(grp)
        n = len(season_rows)

        for target_idx in range(seq_len, n):
            window = season_rows[target_idx - seq_len: target_idx]
            target = season_rows[target_idx]

            sequences.append({
                "window": window,
                # Target game's feature row appended as K+1th context token.
                # Contains: rolling stats from T-1, season avgs up to T-1,
                # plus target game context (home/away, opponent, def stats, rest).
                "context_token": target,
                "target_points": target["target_points"],
                "target_rebounds": target["target_rebounds"],
                "target_assists": target["target_assists"],
                "target_minutes": target["target_minutes"],
                # Metadata for splitting
                "game_date": target["game_date"],
                "season": season,
            })

    # Sort globally by date for time-based split
    sequences.sort(key=lambda s: s["game_date"])
    return sequences


# ---------------------------------------------------------------------------
# Step 6: Encode categoricals & build tensors
# ---------------------------------------------------------------------------

def encode_and_tensorize(
    sequences: list[dict],
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
):
    """
    Fit LabelEncoders on train split, encode all sequences, fit StandardScaler
    on train continuous features, return train/val/test tensor dicts + meta.

    Split order (chronological): [---- train ----][-- val --][- test -]
    The test set is the most recent data and is never used during training.
    Encoders and scaler are fit on train only, then applied to val and test.
    """
    n = len(sequences)
    test_idx = int(n * (1 - test_frac))
    val_idx = int(n * (1 - test_frac - val_frac))
    train_seqs = sequences[:val_idx]
    val_seqs = sequences[val_idx:test_idx]
    test_seqs = sequences[test_idx:]

    print(
        f"  Train: {len(train_seqs):,}  |  Val: {len(val_seqs):,}  |  Test: {len(test_seqs):,}"
        f"  (held out — not used during training)"
    )

    # Fit label encoders on train set
    # Include context_token rows so OOV teams/players in the context token are handled
    def collect_ids(seqs, field):
        return (
            [row[field] for seq in seqs for row in seq["window"]]
            + [seq["context_token"][field] for seq in seqs]
        )

    person_enc = LabelEncoder()
    team_enc = LabelEncoder()

    all_train_pids = collect_ids(train_seqs, "person_id")
    all_train_teams = (
        collect_ids(train_seqs, "player_team_name")
        + collect_ids(train_seqs, "opponent_team_name")
    )
    person_enc.fit(all_train_pids)
    team_enc.fit(all_train_teams)

    num_players = len(person_enc.classes_)
    num_teams = len(team_enc.classes_)
    print(f"  Unique players (train): {num_players}  |  Unique teams (train): {num_teams}")

    TOK = SEQ_LEN + 1  # tokens per sequence (K past games + 1 context)

    def fill_arrays(seqs, label: str, scaler=None):
        """
        Pre-allocate numpy arrays and fill them in one pass.
        If scaler is None, fit a new StandardScaler on this split's data.
        Scales in CHUNK-sized blocks to keep peak memory low.
        Returns (tensor_dict, fitted_scaler).
        """
        n = len(seqs)
        CHUNK = 5_000

        # Pre-allocate — memory is reserved once, no intermediate list needed
        x_cont   = np.empty((n, TOK, NUM_CONT), dtype=np.float32)
        x_pid    = np.empty(n, dtype=np.int64)
        x_pt     = np.empty((n, TOK), dtype=np.int64)
        x_ot     = np.empty((n, TOK), dtype=np.int64)
        x_gi     = np.empty((n, TOK), dtype=np.int64)
        y        = np.empty((n, 4), dtype=np.float32)

        print(f"  Filling {label} arrays ({n:,} sequences)...")
        for i, seq in enumerate(seqs):
            w   = seq["window"]
            ctx = seq["context_token"]
            tok = w + [ctx]

            x_cont[i] = [[row[f] for f in CONT_FEATURE_NAMES] for row in tok]

            try:
                x_pid[i] = int(person_enc.transform([w[0]["person_id"]])[0])
            except ValueError:
                x_pid[i] = 0
            try:
                x_pt[i] = team_enc.transform([r["player_team_name"] for r in tok])
            except ValueError:
                x_pt[i] = 0
            try:
                x_ot[i] = team_enc.transform([r["opponent_team_name"] for r in tok])
            except ValueError:
                x_ot[i] = 0

            x_gi[i] = [min(r["game_idx"], 82) for r in tok]
            y[i] = [seq["target_points"], seq["target_assists"],
                    seq["target_rebounds"], seq["target_minutes"]]

            if (i + 1) % 100_000 == 0:
                print(f"    {i + 1:,} / {n:,}")

        # Fit scaler on train (using a flat view — no extra copy)
        if scaler is None:
            flat = x_cont.reshape(-1, NUM_CONT)  # view, no copy
            scaler = StandardScaler()
            scaler.fit(flat)
            print(f"  Scaler fitted on {flat.shape[0]:,} timestep rows.")

        # Scale in-place chunk by chunk (only one chunk extra in memory at a time)
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            chunk = x_cont[start:end].reshape(-1, NUM_CONT)
            x_cont[start:end] = scaler.transform(chunk).reshape(end - start, TOK, NUM_CONT)

        tensors = {
            "x_cont":        torch.from_numpy(x_cont),
            "x_person_id":   torch.from_numpy(x_pid),
            "x_player_team": torch.from_numpy(x_pt),
            "x_opp_team":    torch.from_numpy(x_ot),
            "x_game_idx":    torch.from_numpy(x_gi),
            "y":             torch.from_numpy(y),
        }
        return tensors, scaler

    train_tensors, scaler = fill_arrays(train_seqs, "train")
    val_tensors,   _      = fill_arrays(val_seqs,   "val",  scaler=scaler)
    test_tensors,  _      = fill_arrays(test_seqs,  "test (held out)", scaler=scaler)

    meta = {
        "person_enc": person_enc,
        "team_enc": team_enc,
        "scaler": scaler,
        "num_players": num_players,
        "num_teams": num_teams,
        "num_cont_features": NUM_CONT,
        "cont_feature_names": CONT_FEATURE_NAMES,
        "seq_len": SEQ_LEN + 1,  # K past games + 1 target context token
    }
    return train_tensors, val_tensors, test_tensors, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_players: Optional[int], val_frac: float, test_frac: float, min_games: int):
    engine = create_engine(DATABASE_URL)

    print("=== Step 1: Loading player games ===")
    df = load_player_games(engine, min_games, max_players)

    print("\n=== Step 2: Computing team game totals ===")
    team_totals = compute_team_game_totals(df)

    print("=== Step 3: Building team defensive lookup (this may take a minute) ===")
    team_def_lookup = build_team_defensive_lookup(df, team_totals)
    print(f"  Lookup entries: {len(team_def_lookup):,}")

    print("\n=== Step 4: Computing season averages ===")
    season_avgs = compute_season_averages(df)
    global_means = compute_global_stat_means(season_avgs)

    print("\n=== Step 5: Building per-player feature rows ===")
    all_rows = []
    for pid, player_df in df.groupby("person_id"):
        player_rows = build_player_feature_rows(
            player_df.sort_values("game_date").reset_index(drop=True),
            season_avgs,
            global_means,
            team_def_lookup,
        )
        all_rows.extend(player_rows)
    print(f"  Total feature rows: {len(all_rows):,}")

    print("\n=== Step 6: Building K=10 sliding window sequences ===")
    sequences = build_sequences(all_rows)
    print(f"  Total sequences: {len(sequences):,}")

    print("\n=== Step 7: Encoding categoricals, scaling, building tensors ===")
    train_tensors, val_tensors, test_tensors, meta = encode_and_tensorize(
        sequences, val_frac, test_frac
    )

    print("\n=== Saving outputs ===")
    torch.save(train_tensors, "train.pt")
    torch.save(val_tensors, "val.pt")
    torch.save(test_tensors, "test.pt")
    with open("pipeline_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"  train.pt  → {train_tensors['y'].shape[0]:,} samples")
    print(f"  val.pt    → {val_tensors['y'].shape[0]:,} samples")
    print(f"  test.pt   → {test_tensors['y'].shape[0]:,} samples  (held out)")
    print("  pipeline_meta.pkl saved.")
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-players", type=int, default=None,
                        help="Cap number of players (default: all eligible)")
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC,
                        help="Fraction of data for validation (default: 0.20)")
    parser.add_argument("--test-frac", type=float, default=TEST_FRAC,
                        help="Fraction of data for test set, held out completely (default: 0.05)")
    parser.add_argument("--min-games", type=int, default=MIN_GAMES_PER_PLAYER,
                        help="Min games per player (default: 25)")
    args = parser.parse_args()
    main(args.max_players, args.val_frac, args.test_frac, args.min_games)

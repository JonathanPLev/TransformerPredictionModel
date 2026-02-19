from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.server.config import PLAYER_STATS_CSV, GAMES_CSV, PLAYERS_CSV


SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9\s\-\.\']+")

def clean_query(s: str) -> str:
    s = (s or "").strip()
    s = SAFE_NAME_RE.sub("", s)   # remove unsafe characters
    s = re.sub(r"\s+", " ", s)    # collapse spaces
    return s.strip()

def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path, low_memory=False)

def _add_opp_id(df_stats: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    need = ["gameId", "hometeamId", "awayteamId"]
    for c in need:
        if c not in df_games.columns:
            raise ValueError(f"Games.csv missing required column: {c}")
    if "gameId" not in df_stats.columns or "home" not in df_stats.columns:
        raise ValueError("PlayerStatistics.csv missing required columns: gameId/home")

    g = df_games[need].copy()
    out = df_stats.merge(g, on="gameId", how="left")
    out["home"] = pd.to_numeric(out["home"], errors="coerce").fillna(0).astype(int)

    out["opp_id"] = np.where(out["home"] == 1, out["awayteamId"], out["hometeamId"])
    out["opp_id"] = out["opp_id"].fillna(-1).astype(int).astype(str)
    return out

def _lookup_person_id_by_name(name: str) -> Optional[int]:
    """
    Uses local Players.csv if present.
    Expected columns (common patterns):
      - personId and displayFirstLast OR firstName/lastName
    If file/columns not present -> return None.
    """
    if not PLAYERS_CSV.exists():
        return None

    dfp = _load_csv(PLAYERS_CSV)

    # normalize
    name_norm = name.lower().strip()

    if "displayFirstLast" in dfp.columns and "personId" in dfp.columns:
        dfp["__n"] = dfp["displayFirstLast"].astype(str).str.lower().str.strip()
        hit = dfp[dfp["__n"] == name_norm]
        if not hit.empty:
            return int(hit.iloc[0]["personId"])

        # small fuzzy fallback (no extra deps): substring contains
        hit2 = dfp[dfp["__n"].str.contains(name_norm, na=False)]
        if len(hit2) == 1:
            return int(hit2.iloc[0]["personId"])
        return None

    # if first/last name columns exist
    if {"firstName", "lastName", "personId"}.issubset(dfp.columns):
        dfp["__n"] = (dfp["firstName"].astype(str) + " " + dfp["lastName"].astype(str)).str.lower().str.strip()
        hit = dfp[dfp["__n"] == name_norm]
        if not hit.empty:
            return int(hit.iloc[0]["personId"])
        hit2 = dfp[dfp["__n"].str.contains(name_norm, na=False)]
        if len(hit2) == 1:
            return int(hit2.iloc[0]["personId"])
        return None

    return None

def fetch_player_df(query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """
    status_code:
      200 success
      404 not found / unsupported
      422 injured (stub)
    """
    cleaned = clean_query(query)
    meta: Dict[str, Any] = {"cleaned_query": cleaned}

    # --- injury stub (until real injury integration) ---
    # Example convention: user sends "LeBron James (injured)" or "injured: LeBron James"
    if "injured" in cleaned.lower():
        meta["reason"] = "injured (stub)"
        return None, meta, 422

    # parse as personId if digits
    person_id: Optional[int] = None
    if cleaned.isdigit():
        person_id = int(cleaned)
    else:
        # name lookup locally (Players.csv)
        person_id = _lookup_person_id_by_name(cleaned)
        if person_id is None:
            meta["reason"] = "name lookup not wired (need Players.csv or DB); send numeric personId"
            return None, meta, 404

    # load datasets
    try:
        df_stats = _load_csv(PLAYER_STATS_CSV)
        df_games = _load_csv(GAMES_CSV)
    except Exception as e:
        meta["reason"] = str(e)
        return None, meta, 404

    if "personId" not in df_stats.columns:
        meta["reason"] = "PlayerStatistics.csv missing personId"
        return None, meta, 404

    df_player = df_stats[df_stats["personId"] == person_id].copy()
    if df_player.empty:
        meta["reason"] = f"no rows for personId={person_id}"
        return None, meta, 404

    try:
        df_player = _add_opp_id(df_player, df_games)
    except Exception as e:
        meta["reason"] = f"opp_id merge failed: {e}"
        return None, meta, 404

    meta["person_id"] = person_id
    return df_player, meta, 200
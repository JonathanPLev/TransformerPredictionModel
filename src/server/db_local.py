from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.server.config import PLAYER_STATS_CSV, GAMES_CSV, PLAYERS_CSV


# Allowlist: letters, spaces, hyphen, apostrophe
SAFE_NAME_RE = re.compile(r"[^a-zA-Z\s\-']+")
CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F-\x9F]+")
MAX_NAME_LEN = 64


def clean_query(s: str) -> str:
    """
    Normalize a player-name query:
      - strip leading/trailing whitespace
      - remove control characters
      - collapse internal whitespace
      - enforce max length
      - drop any characters outside [letters, space, hyphen, apostrophe]

    Returns the cleaned string (possibly empty if everything was stripped).
    """
    s = (s or "").strip()
    if not s:
        return ""

    # remove control characters
    s = CONTROL_CHARS_RE.sub("", s)

    # collapse whitespace early to make the length check more predictable
    s = re.sub(r"\s+", " ", s).strip()

    # enforce max length
    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN].strip()

    # remove any characters outside our allowlist
    s = SAFE_NAME_RE.sub("", s)

    # final collapse + strip
    s = re.sub(r"\s+", " ", s).strip()
    return s

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


def _row_to_canonical_name(row: pd.Series) -> str:
    """
    Prefer displayFirstLast if present; otherwise firstName + ' ' + lastName.
    """
    if "displayFirstLast" in row:
        val = str(row["displayFirstLast"]).strip()
        if val:
            return val
    first = str(row.get("firstName", "")).strip()
    last = str(row.get("lastName", "")).strip()
    return f"{first} {last}".strip()


def _lookup_person_id_by_name(name: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Uses local Players.csv if present.
    Expected columns (common patterns):
      - personId and displayFirstLast OR firstName/lastName
    If file/columns not present -> return (None, None).
    """
    if not PLAYERS_CSV.exists():
        return None, None

    dfp = _load_csv(PLAYERS_CSV)

    # normalize
    name_norm = name.lower().strip()

    if "displayFirstLast" in dfp.columns and "personId" in dfp.columns:
        dfp["__n"] = dfp["displayFirstLast"].astype(str).str.lower().str.strip()
        hit = dfp[dfp["__n"] == name_norm]
        if not hit.empty:
            row = hit.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)

        # small fuzzy fallback (no extra deps): substring contains
        hit2 = dfp[dfp["__n"].str.contains(name_norm, na=False)]
        if len(hit2) == 1:
            row = hit2.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)
        return None, None

    # if first/last name columns exist
    if {"firstName", "lastName", "personId"}.issubset(dfp.columns):
        dfp["__n"] = (dfp["firstName"].astype(str) + " " + dfp["lastName"].astype(str)).str.lower().str.strip()
        hit = dfp[dfp["__n"] == name_norm]
        if not hit.empty:
            row = hit.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)
        hit2 = dfp[dfp["__n"].str.contains(name_norm, na=False)]
        if len(hit2) == 1:
            row = hit2.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)
        return None, None

    return None, None


def _lookup_name_by_person_id(person_id: int) -> Optional[str]:
    """
    Best-effort canonical name lookup by personId using Players.csv.
    """
    if not PLAYERS_CSV.exists():
        return None

    dfp = _load_csv(PLAYERS_CSV)
    if "personId" not in dfp.columns:
        return None

    hit = dfp[dfp["personId"] == person_id]
    if hit.empty:
        return None

    row = hit.iloc[0]
    name = _row_to_canonical_name(row)
    return name or None


def fetch_player_df(query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """
    status_code:
      200 success
      404 not found / unsupported
    """
    cleaned = clean_query(query)
    meta: Dict[str, Any] = {"cleaned_query": cleaned}

    if not cleaned:
        meta["reason"] = "empty_or_invalid_player_name"
        return None, meta, 404

    # parse as personId if digits
    person_id: Optional[int] = None
    player_name: Optional[str] = None
    if cleaned.isdigit():
        person_id = int(cleaned)
        player_name = _lookup_name_by_person_id(person_id)
    else:
        # name lookup locally (Players.csv)
        person_id, player_name = _lookup_person_id_by_name(cleaned)
        if person_id is None:
            meta["reason"] = "name_lookup_failed (need Players.csv or DB); send numeric personId"
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
    if player_name:
        meta["player_name"] = player_name
    return df_player, meta, 200
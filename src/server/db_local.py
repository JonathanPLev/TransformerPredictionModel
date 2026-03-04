from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import sqlalchemy as sqla

from src.server.config import DATABASE_URL, PLAYERS_CSV

engine = sqla.create_engine(DATABASE_URL)

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

    s = CONTROL_CHARS_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) > MAX_NAME_LEN:
        s = s[:MAX_NAME_LEN].strip()

    s = SAFE_NAME_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_players_csv() -> pd.DataFrame:
    if not PLAYERS_CSV.exists():
        raise FileNotFoundError(f"Missing required CSV: {PLAYERS_CSV}")
    return pd.read_csv(PLAYERS_CSV, low_memory=False)


def _row_to_canonical_name(row: pd.Series) -> str:
    if "displayFirstLast" in row:
        val = str(row["displayFirstLast"]).strip()
        if val:
            return val
    first = str(row.get("firstName", "")).strip()
    last = str(row.get("lastName", "")).strip()
    return f"{first} {last}".strip()


def _lookup_person_id_by_name(name: str) -> Tuple[Optional[int], Optional[str]]:
    """
    Uses local Players.csv for canonical name resolution.
    Returns (person_id, canonical_name) or (None, None).
    """
    if not PLAYERS_CSV.exists():
        return None, None

    dfp = _load_players_csv()
    name_norm = name.lower().strip()

    if "displayFirstLast" in dfp.columns and "personId" in dfp.columns:
        dfp["__n"] = dfp["displayFirstLast"].astype(str).str.lower().str.strip()
        hit = dfp[dfp["__n"] == name_norm]
        if not hit.empty:
            row = hit.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)

        hit2 = dfp[dfp["__n"].str.contains(name_norm, na=False)]
        if len(hit2) == 1:
            row = hit2.iloc[0]
            return int(row["personId"]), _row_to_canonical_name(row)
        return None, None

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
    if not PLAYERS_CSV.exists():
        return None

    dfp = _load_players_csv()
    if "personId" not in dfp.columns:
        return None

    hit = dfp[dfp["personId"] == person_id]
    if hit.empty:
        return None

    return _row_to_canonical_name(hit.iloc[0]) or None


_PLAYER_TEAM_SQL = sqla.text("""
    SELECT player_team_name
    FROM player_statistics
    WHERE person_id = :pid
      AND game_type = 'Regular Season'
      AND game_date IS NOT NULL
      AND player_team_name IS NOT NULL
    ORDER BY game_date DESC
    LIMIT 1
""")


def get_player_team_name(person_id: int) -> Optional[str]:
    try:
        with engine.connect() as conn:
            row = conn.execute(_PLAYER_TEAM_SQL, {"pid": person_id}).fetchone()
            return row[0] if row else None
    except Exception:
        return None


_MISSED_GAMES_SQL = sqla.text("""
    WITH player_last AS (
        SELECT MAX(game_date) AS last_played
        FROM player_statistics
        WHERE person_id = :pid
          AND game_date IS NOT NULL
    ),
    team_games AS (
        SELECT DISTINCT game_date
        FROM player_statistics
        WHERE player_team_name = :team
          AND game_date IS NOT NULL
          AND game_date > (SELECT last_played FROM player_last)
          AND game_type = 'Regular Season'
    )
    SELECT
        (SELECT last_played FROM player_last) AS last_played,
        COUNT(*) AS missed
    FROM team_games
""")


def check_missed_games(person_id: int, team_name: str) -> Tuple[int, Optional[str]]:
    """Return (games_missed_since_last_appearance, last_played_date_str)."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                _MISSED_GAMES_SQL, {"pid": person_id, "team": team_name}
            ).fetchone()
            if row is None:
                return 0, None
            last_played = str(row[0]) if row[0] else None
            missed = int(row[1]) if row[1] else 0
            return missed, last_played
    except Exception:
        return 0, None


_PLAYER_STATS_SQL = sqla.text("""
    SELECT
        ps.person_id   AS "personId",
        ps.game_id     AS "gameId",
        ps.game_date   AS "gameDate",
        ps.home::int,
        ps.points,
        ps.num_minutes AS "numMinutes",
        ps.assists,
        ps.rebounds_total       AS "reboundsTotal",
        ps.turnovers,
        ps.field_goals_attempted   AS "fieldGoalsAttempted",
        ps.three_pointers_attempted AS "threePointersAttempted",
        ps.free_throws_attempted   AS "freeThrowsAttempted",
        ps.steals,
        ps.blocks,
        ps.plus_minus_points    AS "plusMinusPoints",
        CASE
            WHEN ps.home
                THEN COALESCE(NULLIF(gr.awayteamid,'')::BIGINT, -1)::TEXT
            ELSE COALESCE(NULLIF(gr.hometeamid,'')::BIGINT, -1)::TEXT
        END AS opp_id
    FROM player_statistics ps
    LEFT JOIN (
        SELECT DISTINCT ON (NULLIF(gameid,'')::REAL::INT)
            NULLIF(gameid,'')::REAL::INT AS gid,
            hometeamid,
            awayteamid
        FROM games_raw
        WHERE NULLIF(gameid,'') IS NOT NULL
        ORDER BY NULLIF(gameid,'')::REAL::INT
    ) gr ON ps.game_id = gr.gid
    WHERE ps.person_id = :pid
    ORDER BY ps.game_date ASC
""")


def fetch_player_df(query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], int]:
    """
    Resolve a player name, query the PostgreSQL database for their stats,
    and return a DataFrame ready for service.py prediction.

    status_code: 200 success, 404 not found / error
    """
    cleaned = clean_query(query)
    meta: Dict[str, Any] = {"cleaned_query": cleaned}

    if not cleaned:
        meta["reason"] = "empty_or_invalid_player_name"
        return None, meta, 404

    person_id: Optional[int] = None
    player_name: Optional[str] = None
    if cleaned.isdigit():
        person_id = int(cleaned)
        player_name = _lookup_name_by_person_id(person_id)
    else:
        person_id, player_name = _lookup_person_id_by_name(cleaned)
        if person_id is None:
            meta["reason"] = "Player not found"
            return None, meta, 404

    try:
        with engine.connect() as conn:
            result = conn.execute(_PLAYER_STATS_SQL, {"pid": person_id})
            df_player = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        meta["reason"] = f"database error: {e}"
        return None, meta, 404

    if df_player.empty:
        meta["reason"] = f"no rows for personId={person_id}"
        return None, meta, 404

    meta["person_id"] = person_id
    if player_name:
        meta["player_name"] = player_name
    return df_player, meta, 200

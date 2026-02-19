from __future__ import annotations

import os
from typing import Any

DATABASE_URL = os.environ.get("DATABASE_URL")
DB_ENABLED = os.environ.get("DB_ENABLED", "").lower() in ("1", "true", "yes")


def execute_parameterized(
    query: str,
    params: tuple[Any, ...],
    *,
    fetch: bool = True,
) -> list[tuple[Any, ...]] | None:

    if not DB_ENABLED or not DATABASE_URL:
        return None

    try:
        import psycopg2
    except ImportError:
        return None

    try:
        with psycopg2.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)  
                if fetch:
                    return cur.fetchall()
                return []
    except Exception:
        return None


def get_next_game_by_player_name(cleaned_player_name: str) -> dict[str, Any] | None:
    if not cleaned_player_name or not isinstance(cleaned_player_name, str):
        return None

    #placeholder query
    query = """
        SELECT game_ts_est, opponent_abbr
        FROM schedule
        WHERE player_name = %s AND game_ts_est >= NOW()
        ORDER BY game_ts_est ASC
        LIMIT 1
    """
    params = (cleaned_player_name,)
    rows = execute_parameterized(query, params, fetch=True)
    if not rows or not rows[0]:
        return None

    row = rows[0]
    return {
        "time_and_date_est": str(row[0]) if row[0] is not None else None,
        "team_against": str(row[1]) if row[1] is not None else None,
    }

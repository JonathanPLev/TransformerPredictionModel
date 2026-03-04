from __future__ import annotations

import threading
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from nba_api.stats.static import players as nba_players
from src.training_player_data.fetch_player_data import injury_status

from src.server.config import MODEL_PATH, PREPROC_PATH
from src.server.model_loader import load_model_and_preproc
from src.server.db_local import fetch_player_df, get_player_team_name, check_missed_games
from src.server.service import predict_next_points
from src.training_player_data.fetch_player_data import schedule_status

INJURED_STATUSES = {"out", "out for season", "inactive"}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )

# Cached injury report keyed by 30-min bucket to avoid PDF file-lock
# issues from concurrent nbainjuries calls.
_injury_lock = threading.Lock()
_injury_cache: dict = {"timestamp_key": None, "report": None}


def _get_cached_injury_status(first_name: str, last_name: str):
    """
    Thin caching wrapper around fetch_player_data.injury_status.
    Caches the full report DataFrame for the current 30-minute window
    so only the first request in each window hits the PDF.
    """
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    minute = 30 if now.minute >= 30 else 0
    ts_key = now.replace(minute=minute, second=0, microsecond=0)

    with _injury_lock:
        if _injury_cache["timestamp_key"] == ts_key and _injury_cache["report"] is not None:
            cached = _injury_cache["report"]
            return _match_player_in_report(cached, first_name, last_name)

    result = injury_status(first_name, last_name)

    return result


def _match_player_in_report(report, first_name: str, last_name: str):
    """Look up a single player in an already-loaded injury report DataFrame."""
    if report is None or report.empty:
        return None
    needle = f"{last_name.strip()}, {first_name.strip()}".lower()
    names = report["Player Name"].astype(str).str.strip().str.lower()
    match = report[names == needle]
    return match if not match.empty else None


def check_player_availability(
    display_name: str, has_db_data: bool = False
) -> tuple[bool, str | None]:
    found = nba_players.find_players_by_full_name(display_name)
    if not found:
        return False, None

    player = found[0]

    if player.get("is_active") is False and not has_db_data:
        return True, "Player is not currently active in the NBA"

    first_name = player.get("first_name", "")
    last_name = player.get("last_name", "")
    if first_name and last_name:
        try:
            inj_df = injury_status(
                _strip_accents(first_name), _strip_accents(last_name)
            )
            if inj_df is not None and not inj_df.empty:
                status = str(inj_df.iloc[0].get("Current Status", "")).strip()
                print(f"  [injury] {display_name} -> status={status!r}")
                if status.lower() in INJURED_STATUSES:
                    return True, f"Player is currently injured ({status})"
        except Exception as exc:
            print(f"  [injury] error checking {display_name}: {exc}")

    return False, None


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


class PredictRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    model, preproc = load_model_and_preproc(
        model_path=str(MODEL_PATH),
        preproc_path=str(PREPROC_PATH),
    )
    app.state.model = model
    app.state.preproc = preproc
    app.state.ready = True

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    if getattr(app.state, "ready", False):
        return {"status": "ready"}
    return JSONResponse(status_code=503, content={"status": "not_ready"})

@app.post("/predict")
def predict(req: PredictRequest):
    # default response shell
    base = {
        "prediction": None,
        "metadata": {"cleaned_query": req.query},
        "errors": [],
        "warnings": [],
        "status_code": 200,
    }

    df, meta, status_code = fetch_player_df(req.query)
    base["metadata"].update(meta or {})

    # pass-through status codes from fetch layer
    if status_code != 200:
        base["status_code"] = status_code
        return JSONResponse(status_code=status_code, content=base)

    has_recent_data = False
    if df is not None and "gameDate" in df.columns:
        try:
            import pandas as pd
            latest = pd.to_datetime(df["gameDate"], errors="coerce").max()
            if pd.notna(latest):
                latest = latest.tz_localize(None) if latest.tzinfo else latest
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=365)
                has_recent_data = latest > cutoff
                print(f"  [recency] latest game={latest.date()}, cutoff={cutoff.date()}, recent={has_recent_data}")
        except Exception:
            pass

    player_name = base["metadata"].get("player_name") or base["metadata"].get("cleaned_query")
    unavailable, reason = check_player_availability(player_name or "", has_db_data=has_recent_data)
    print(f"[availability] player={player_name!r}  unavailable={unavailable}  reason={reason!r}")
    if unavailable:
        base["status_code"] = 422
        is_inactive = reason and "not currently active" in reason
        base["metadata"]["error_type"] = "inactive" if is_inactive else "injured"
        return JSONResponse(status_code=422, content=base)

    # --- missed-games check (catches long-term injuries not on today's report) ---
    MISSED_GAMES_THRESHOLD = 3
    person_id = base["metadata"].get("person_id")
    if person_id:
        team_name = get_player_team_name(person_id)
        if team_name:
            missed, last_played = check_missed_games(person_id, team_name)
            print(f"[missed-games] {player_name}: missed={missed}, last_played={last_played}")
            if missed >= MISSED_GAMES_THRESHOLD:
                base["status_code"] = 422
                base["metadata"]["error_type"] = "injured"
                return JSONResponse(status_code=422, content=base)

    # --- schedule lookup (opponent + game time) ---
    if person_id:
        try:
            if not team_name:
                team_name = get_player_team_name(person_id)
            if team_name:
                sched_df = df.rename(columns={"gameDate": "game_date"})
                sched = schedule_status(sched_df, team_name)
                if sched:
                    base["metadata"]["opponent"] = sched["opponent"]
                    try:
                        from datetime import datetime as _dt
                        raw_ts = sched["next_game_ts_est"]
                        parsed = _dt.fromisoformat(str(raw_ts))
                        base["metadata"]["game_datetime_est"] = parsed.strftime("%b %#d at %#I:%M %p EST")
                    except Exception:
                        base["metadata"]["game_datetime_est"] = sched["next_game_ts_est"]
        except Exception as exc:
            print(f"  [schedule] error: {exc}")

    # --- prediction ---
    try:
        pred = predict_next_points(df, app.state.model, app.state.preproc, window=5)
        base["prediction"] = float(pred)
        base["status_code"] = 200

        display_name = base["metadata"].get("player_name") or base["metadata"].get("cleaned_query")
        opp = base["metadata"].get("opponent")
        game_time_est = base["metadata"].get("game_datetime_est")
        if display_name and opp and game_time_est:
            log_line = (
                f"Prediction result: {display_name} will score {pred:.1f} points "
                f"during their next game against {opp} on {game_time_est} (EST)"
            )
        elif display_name:
            log_line = f"Prediction result: {display_name} will score {pred:.1f} points in their next game."
        else:
            log_line = f"Prediction result: player will score {pred:.1f} points in their next game."
        print(log_line)

        return JSONResponse(status_code=200, content=base)
    except Exception as e:
        print(f"[predict] error: {e}")
        base["status_code"] = 404
        return JSONResponse(status_code=404, content=base)

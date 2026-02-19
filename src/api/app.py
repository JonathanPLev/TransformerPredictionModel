from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from utils.input_validation import sanitize_player_name
from utils.shared_utils import PlayerUtils
from api.schemas import PredictRequest, PredictSuccessResponse, PredictErrorResponse

_src_dir = Path(__file__).resolve().parent.parent
_dataset_path = str(_src_dir / "data_generation" / "NBA_Multi_Player_Training_Data.csv")
_mapping_path = str(_src_dir / "data_generation" / "player_id_mapping.csv")
_demo_artifact_dir = _src_dir.parent / "models" / "training_player_data"


def _get_rf_predictor():
    from random_forest.multi_player_predictor import MultiPlayerPredictor
    return MultiPlayerPredictor(dataset_path=_dataset_path, mapping_path=_mapping_path)


def _predict_demo(person_id: int) -> float | None:
    """Use train_one_player demo model (saved with --save). Parameterized/CSV only; no raw input in DB."""
    from training_player_data.train_one_player import predict_next_points_by_person_id
    return predict_next_points_by_person_id(person_id, artifact_dir=_demo_artifact_dir)


app = FastAPI(title="Player Prediction API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/api/predict",
    response_model=PredictSuccessResponse,
    responses={
        200: {"description": "Success", "model": PredictSuccessResponse},
        404: {"description": "Player not found", "model": PredictErrorResponse},
        422: {"description": "Validation / injury", "model": PredictErrorResponse},
    },
)
def predict(request: PredictRequest):
    """
    Accept sanitized player name only. Backend re-validates (defense in depth).
    Never uses raw input in DB; any DB access uses parameterized queries only.
    """
    raw = request.playerName

    # Sanitize again (defense in depth; frontend also cleans—keep both for direct API safety).
    try:
        cleaned = sanitize_player_name(raw)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"errorMessage": str(e)},
        )

    try:
        from nba_api.stats.static import players
        matches = players.find_players_by_full_name(cleaned)
        if not matches:
            return JSONResponse(
                status_code=404,
                content={"errorMessage": "Player not found"},
            )
        official_name = matches[0].get("full_name") or cleaned
        player_id = matches[0]["id"]
        team_abbr = PlayerUtils.get_player_team(player_id)
    except Exception as e:
        if "not found" in str(e).lower() or "no player" in str(e).lower():
            return JSONResponse(
                status_code=404,
                content={"errorMessage": "Player not found"},
            )
        return JSONResponse(
            status_code=422,
            content={"errorMessage": str(e)},
        )

    time_and_date_est = None  
    team_against_from_db = None

    demo_pts = _predict_demo(player_id)
    if demo_pts is not None:
        return PredictSuccessResponse(
            playerResult=float(demo_pts),
            officialPlayerName=official_name,
            teamAgainst=team_against_from_db,
            timeAndDateEST=time_and_date_est,
        )

  
    predictor = _get_rf_predictor()
    if predictor.data is None:
        return JSONResponse(
            status_code=503,
            content={"errorMessage": "Prediction service unavailable (no dataset). For demo, run: python -m training_player_data.train_one_player --person_id <id> --save"},
        )
    if not predictor.is_trained:
        try:
            predictor.train_model()
            predictor.save_model()
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={"errorMessage": f"Prediction service unavailable: {e}"},
            )

    result = predictor.predict_for_player(player_name=cleaned)
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"errorMessage": "Player not found"},
        )

        return PredictSuccessResponse(
            playerResult=result.get("over_probability"),
            officialPlayerName=official_name,
            teamAgainst=result.get("opponent"),
            timeAndDateEST=time_and_date_est,
        )


@app.get("/health")
def health():
    return {"status": "ok"}

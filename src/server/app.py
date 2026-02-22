from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from src.server.config import MODEL_PATH, PREPROC_PATH
from src.server.model_loader import load_model_and_preproc
from src.server.db_local import fetch_player_df
from src.server.service import predict_next_points

app = FastAPI()

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
        base["errors"].append(meta.get("reason", "error") if meta else "error")
        return JSONResponse(status_code=status_code, content=base)

    # prediction
    try:
        pred = predict_next_points(df, app.state.model, app.state.preproc, window=5)
        base["prediction"] = float(pred)
        base["status_code"] = 200
        return JSONResponse(status_code=200, content=base)
    except Exception as e:
        # per your requirement: any error should be 404
        base["status_code"] = 404
        base["errors"].append(f"Predict error: {e}")
        return JSONResponse(status_code=404, content=base)
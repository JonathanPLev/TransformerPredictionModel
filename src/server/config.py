from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root

MODEL_PATH = ROOT / "model_state.pt"
PREPROC_PATH = ROOT / "preproc.joblib"

DATABASE_URL = "postgresql+psycopg2://nba:nba@172.24.196.46:5432/nba"

DATA_DIR = ROOT / "src" / "nba_dataset" / "Data"
PLAYERS_CSV = DATA_DIR / "Players.csv"

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root

MODEL_PATH = ROOT / "model_state.pt"
PREPROC_PATH = ROOT / "preproc.joblib"

DATA_DIR = ROOT / "src" / "nba_dataset" / "Data"
PLAYER_STATS_CSV = DATA_DIR / "PlayerStatistics.csv"
GAMES_CSV = DATA_DIR / "Games.csv"

# Optional (if your dataset has it). If not present, name lookup will fall back to 404.
PLAYERS_CSV = DATA_DIR / "Players.csv"
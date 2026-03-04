import numpy as np
import pandas as pd
import torch

def _pick_date_col(df: pd.DataFrame) -> str:
    for c in ["gameDate", "gameDateTimeEst", "gameDateTimeUTC", "gameDateTime", "gameDateTimeLocal"]:
        if c in df.columns:
            return c
    raise ValueError("No known date column found in df")

# Must match train_one_player.py: LAG_K_DEFAULT=5 and same lag_base
LAG_K = 5


def _build_last_row(df: pd.DataFrame, window: int) -> pd.DataFrame:
    base_feats = [
        "points", "numMinutes", "assists", "reboundsTotal", "turnovers",
        "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted",
        "steals", "blocks", "plusMinusPoints",
    ]
    lag_base = ["reboundsTotal", "assists", "numMinutes"]

    if "opp_id" not in df.columns:
        raise ValueError("df missing opp_id")

    date_col = _pick_date_col(df)
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in base_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # days_rest historical
    df["days_rest"] = df[date_col].diff().dt.days.fillna(0).clip(lower=0)

    feats_present = [c for c in base_feats if c in df.columns]
    roll_df = df[feats_present].rolling(window).mean()
    roll_df.columns = [f"{c}_roll{window}" for c in feats_present]
    df = pd.concat([df, roll_df], axis=1)

    # Lag 1..LAG_K (must match train_one_player.py so preproc columns align)
    lag_present = [c for c in lag_base if c in df.columns]
    lag_dfs = []
    for k in range(1, LAG_K + 1):
        tmp = df[lag_present].shift(k)
        tmp.columns = [f"{c}_lag{k}" for c in lag_present]
        lag_dfs.append(tmp)
    if lag_dfs:
        df = pd.concat([df] + lag_dfs, axis=1)

    roll_cols = [f"{c}_roll{window}" for c in feats_present]
    lag_cols = [f"{c}_lag{k}" for c in lag_present for k in range(1, LAG_K + 1)]
    engineered_cols = roll_cols + lag_cols + ["days_rest", "opp_id"]

    last = df.dropna(subset=engineered_cols).tail(1)
    if last.empty:
        raise ValueError("Not enough valid rows to build features.")
    return last[engineered_cols]

def predict_next_points(df_player: pd.DataFrame, model, preproc, window: int = 5) -> float:
    last_row = _build_last_row(df_player, window=window)
    x = preproc.transform(last_row)
    xt = torch.tensor(x, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_log = model(xt).item()

    pts = float(np.expm1(pred_log))
    return max(0.0, pts)
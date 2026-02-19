import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from torch.utils.data import TensorDataset, DataLoader

# ---------------- config ----------------
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-4
SEED = 42

# ---------------- model ----------------
class MiniNet(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),  # regression output (log-points)
        )

    def forward(self, x):
        return self.net(x)

# ---------------- data loading ----------------
def load_player_stats_csv() -> pd.DataFrame:
    current_dir = Path(__file__).resolve().parent  # .../src/training_player_data
    data_path = current_dir.parent / "nba_dataset" / "Data" / "PlayerStatistics.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find: {data_path}")
    return pd.read_csv(data_path, low_memory=False)

def load_games_csv() -> pd.DataFrame:
    current_dir = Path(__file__).resolve().parent
    games_path = current_dir.parent / "nba_dataset" / "Data" / "Games.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"Could not find: {games_path}")
    return pd.read_csv(games_path, low_memory=False)

def pick_date_col(df: pd.DataFrame) -> str:
    for c in ["gameDate", "gameDateTimeEst", "gameDateTimeUTC", "gameDateTime", "gameDateTimeLocal"]:
        if c in df.columns:
            return c
    raise ValueError(f"No known date column found. Columns include: {list(df.columns)[:60]} ...")

def add_opponent_from_games(df_stats: pd.DataFrame, df_games: pd.DataFrame) -> pd.DataFrame:
    """
    Requires:
      - PlayerStatistics.csv: gameId, home
      - Games.csv: gameId, hometeamId, awayteamId
    Produces:
      - opp_id (string) categorical for one-hot
    """
    needed_games = ["gameId", "hometeamId", "awayteamId"]
    for c in needed_games:
        if c not in df_games.columns:
            raise ValueError(f"Games.csv missing required column: {c}")

    if "gameId" not in df_stats.columns:
        raise ValueError("PlayerStatistics.csv missing required column: gameId")
    if "home" not in df_stats.columns:
        raise ValueError("PlayerStatistics.csv missing required column: home (0/1)")

    g = df_games[needed_games].copy()
    out = df_stats.merge(g, on="gameId", how="left")

    out["home"] = pd.to_numeric(out["home"], errors="coerce").fillna(0).astype(int)

    # Opponent ID: if player is on home team -> opponent is awayteamId; else -> hometeamId
    out["opp_id"] = np.where(out["home"] == 1, out["awayteamId"], out["hometeamId"])
    out["opp_id"] = out["opp_id"].fillna(-1).astype(int).astype(str)

    return out

def build_all_players_dataset(df: pd.DataFrame, window: int):
    if "personId" not in df.columns or "points" not in df.columns:
        raise ValueError("CSV must contain at least 'personId' and 'points'.")

    date_col = pick_date_col(df)

    base_feats = [
        "points",
        "numMinutes",
        "assists",
        "reboundsTotal",
        "turnovers",
        "fieldGoalsAttempted",
        "threePointersAttempted",
        "freeThrowsAttempted",
        "steals",
        "blocks",
        "plusMinusPoints",
    ]

    if "opp_id" not in df.columns:
        raise ValueError("Missing 'opp_id'. Did you merge Games.csv using add_opponent_from_games()?")

    keep = ["personId", "gameId", "home", "opp_id", date_col] + [c for c in base_feats if c in df.columns]
    df = df[keep].copy()

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    for c in base_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    all_rows = []
    numeric_cols = None
    cat_cols = ["opp_id"]

    lag_base = ["reboundsTotal", "assists", "numMinutes"]

    for pid, g in df.groupby("personId", sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)

        # target: next game points
        g["y_next_points"] = g["points"].shift(-1)

        # days of rest
        g["days_rest"] = g[date_col].diff().dt.days.fillna(0).clip(lower=0)

        # ---- BATCH rolling means (fast; avoids pandas fragmentation) ----
        feats_present = [c for c in base_feats if c in g.columns]
        roll_df = g[feats_present].rolling(window).mean()
        roll_df.columns = [f"{c}_roll{window}" for c in feats_present]
        g = pd.concat([g, roll_df], axis=1)

        # ---- BATCH lag-1 features ----
        lag_present = [c for c in lag_base if c in g.columns]
        lag_df = g[lag_present].shift(1)
        lag_df.columns = [f"{c}_lag1" for c in lag_present]
        g = pd.concat([g, lag_df], axis=1)

        roll_cols = [f"{c}_roll{window}" for c in feats_present]
        lag_cols = [f"{c}_lag1" for c in lag_present]
        this_numeric = roll_cols + lag_cols + ["days_rest"]

        if not roll_cols:
            continue

        if numeric_cols is None:
            numeric_cols = this_numeric

        out = g.dropna(subset=this_numeric + ["y_next_points"]).copy()
        if out.empty:
            continue

        out["y_log"] = np.log1p(out["y_next_points"].astype(float))
        all_rows.append(out[["personId", date_col, "opp_id"] + this_numeric + ["y_log"]])

    if not all_rows or numeric_cols is None:
        raise ValueError("No training rows created. Try a smaller window (ex: 3).")

    big = pd.concat(all_rows, ignore_index=True).sort_values(date_col).reset_index(drop=True)
    return big, numeric_cols, cat_cols, date_col

def time_split_df(X_df: pd.DataFrame, y: np.ndarray, train_frac=0.80, val_frac=0.10):
    n = len(X_df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    X_train = X_df.iloc[:n_train].reset_index(drop=True)
    y_train = y[:n_train]

    X_val = X_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    y_val = y[n_train:n_train + n_val]

    X_test = X_df.iloc[n_train + n_val:].reset_index(drop=True)
    y_test = y[n_train + n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------------- training (MINI-BATCH) ----------------
def train_regression_model(X_train_df, y_train, X_val_df, y_val, numeric_cols, cat_cols):
    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )

    X_train = preproc.fit_transform(X_train_df)
    X_val = preproc.transform(X_val_df)

    model = MiniNet(input_size=X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    best_val = float("inf")
    best_state = None
    patience = 0

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mse = criterion(val_pred, y_val_t).item()
                val_mae_log = torch.mean(torch.abs(val_pred - y_val_t)).item()

            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | val_MSE(log) {val_mse:.4f} | val_MAE(log) {val_mae_log:.3f}")

            if patience >= PATIENCE:
                print(f"Early stopping (best val_MSE(log) {best_val:.4f})")
                break

    except KeyboardInterrupt:
        print("\nStopped early (Ctrl+C). Using best model so far...")

    if best_state is None:
        raise RuntimeError("Training ended before any best model was saved. Try running again (or lower the batch size).")

    model.load_state_dict(best_state)
    return model, preproc

def build_last_row_for_player(df_all: pd.DataFrame, person_id: int, date_col: str, window: int,
                              numeric_cols: list, cat_cols: list) -> pd.DataFrame:
    base_feats = [
        "points","numMinutes","assists","reboundsTotal","turnovers",
        "fieldGoalsAttempted","threePointersAttempted","freeThrowsAttempted",
        "steals","blocks","plusMinusPoints"
    ]
    lag_base = ["reboundsTotal", "assists", "numMinutes"]

    one = df_all[df_all["personId"] == person_id].copy()
    if one.empty:
        raise ValueError(f"No rows found for personId={person_id}")

    one[date_col] = pd.to_datetime(one[date_col], utc=True, errors="coerce")
    one = one.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in base_feats:
        if c in one.columns:
            one[c] = pd.to_numeric(one[c], errors="coerce")

    one["days_rest"] = one[date_col].diff().dt.days.fillna(0).clip(lower=0)

    # ---- BATCH rolling + lag (fast) ----
    feats_present = [c for c in base_feats if c in one.columns]
    roll_df = one[feats_present].rolling(window).mean()
    roll_df.columns = [f"{c}_roll{window}" for c in feats_present]
    one = pd.concat([one, roll_df], axis=1)

    lag_present = [c for c in lag_base if c in one.columns]
    lag_df = one[lag_present].shift(1)
    lag_df.columns = [f"{c}_lag1" for c in lag_present]
    one = pd.concat([one, lag_df], axis=1)

    last = one.dropna(subset=numeric_cols + cat_cols).tail(1)
    if last.empty:
        raise ValueError(f"Not enough games for personId={person_id} with window={window}. Try --window 3.")

    return last[numeric_cols + cat_cols]

def predict_next_for_player(df_all: pd.DataFrame, person_id: int, date_col: str, window: int,
                            numeric_cols: list, cat_cols: list, model, preproc) -> float:
    last_row = build_last_row_for_player(df_all, person_id, date_col, window, numeric_cols, cat_cols)
    x = preproc.transform(last_row)
    xt = torch.tensor(x, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        pred_log = model(xt).item()

    pts = float(np.expm1(pred_log))
    return max(0.0, pts)

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    parser = argparse.ArgumentParser()
    parser.add_argument("--person_id", type=int, required=True)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    df_stats = load_player_stats_csv()
    df_games = load_games_csv()

    # add opponent id feature from Games.csv
    df = add_opponent_from_games(df_stats, df_games)
    date_col = pick_date_col(df)

    big, numeric_cols, cat_cols, date_col = build_all_players_dataset(df, window=args.window)

    X_df = big[numeric_cols + cat_cols]
    y = big["y_log"].to_numpy(dtype=np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = time_split_df(X_df, y)

    print(f"Training rows: {len(X_df)} | Numeric: {len(numeric_cols)} | Cat: {len(cat_cols)} | Window: {args.window}")
    model, preproc = train_regression_model(X_train, y_train, X_val, y_val, numeric_cols, cat_cols)

    # Test MAE in real points
    X_test_t = preproc.transform(X_test)
    with torch.no_grad():
        pred_log = model(torch.tensor(X_test_t, dtype=torch.float32)).squeeze(1).numpy()
    pred_pts = np.expm1(pred_log)
    true_pts = np.expm1(y_test)
    test_mae_pts = float(np.mean(np.abs(pred_pts - true_pts)))
    print(f"\nTest MAE (points): {test_mae_pts:.3f}")

    # Baseline: last-N average for that player
    one_points = pd.to_numeric(df[df["personId"] == args.person_id]["points"], errors="coerce").dropna()
    if len(one_points) >= args.window:
        avg_last = float(one_points.tail(args.window).mean())
        print(f"Last {args.window}-game avg (points): {avg_last:.1f}")

    next_pts = predict_next_for_player(df, args.person_id, date_col, args.window, numeric_cols, cat_cols, model, preproc)
    print(f"\nPredicted NEXT game points for personId={args.person_id}: {next_pts:.1f}")

    # save checkpoint
    torch.save(model.state_dict(), "model_state.pt")
    joblib.dump(preproc, "preproc.joblib")
    print("\nSaved checkpoint: model_state.pt + preproc.joblib")

if __name__ == "__main__":
    main()

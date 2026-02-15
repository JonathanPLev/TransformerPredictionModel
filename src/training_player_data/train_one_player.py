import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
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

def pick_date_col(df: pd.DataFrame) -> str:
    for c in ["gameDate", "gameDateTimeEst", "gameDateTimeUTC", "gameDateTime", "gameDateTimeLocal"]:
        if c in df.columns:
            return c
    raise ValueError(f"No known date column found. Columns include: {list(df.columns)[:60]} ...")

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

    keep = ["personId", date_col] + [c for c in base_feats if c in df.columns]
    df = df[keep].copy()

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    for c in base_feats:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    all_rows = []
    feat_cols = None

    for pid, g in df.groupby("personId"):
        g = g.sort_values(date_col).reset_index(drop=True)

        g["y_next_points"] = g["points"].shift(-1)

        for c in base_feats:
            if c in g.columns:
                g[f"{c}_roll{window}"] = g[c].rolling(window).mean()

        this_feat_cols = [f"{c}_roll{window}" for c in base_feats if f"{c}_roll{window}" in g.columns]
        if not this_feat_cols:
            continue
        if feat_cols is None:
            feat_cols = this_feat_cols

        out = g.dropna(subset=this_feat_cols + ["y_next_points"]).copy()
        if out.empty:
            continue

        out["y_log"] = np.log1p(out["y_next_points"].astype(float))
        all_rows.append(out[["personId", date_col] + this_feat_cols + ["y_log"]])

    if not all_rows or feat_cols is None:
        raise ValueError("No training rows created. Try a smaller window (ex: 3).")

    big = pd.concat(all_rows, ignore_index=True)
    big = big.sort_values(date_col).reset_index(drop=True)
    return big, feat_cols, date_col

def time_split(X, y, train_frac=0.80, val_frac=0.10):
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return X_train, y_train, X_val, y_val, X_test, y_test

# ---------------- training (MINI-BATCH) ----------------
def train_regression_model(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = MiniNet(input_size=X_train_s.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
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
    return model, scaler

def predict_next_for_player(df_raw: pd.DataFrame, person_id: int, feat_cols: list, date_col: str, window: int,
                            model, scaler) -> float:
    base = [c.replace(f"_roll{window}", "") for c in feat_cols]

    one = df_raw[df_raw["personId"] == person_id].copy()
    if one.empty:
        raise ValueError(f"No rows found for personId={person_id}")

    one[date_col] = pd.to_datetime(one[date_col], utc=True, errors="coerce")
    one = one.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in base:
        if c in one.columns:
            one[c] = pd.to_numeric(one[c], errors="coerce")
            one[f"{c}_roll{window}"] = one[c].rolling(window).mean()

    last = one.dropna(subset=feat_cols).tail(1)
    if last.empty:
        raise ValueError(f"Not enough games for personId={person_id} with window={window}. Try --window 3.")

    x = last[feat_cols].to_numpy(dtype=np.float32)
    x_s = scaler.transform(x)
    xt = torch.tensor(x_s, dtype=torch.float32)

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

    df = load_player_stats_csv()
    date_col = pick_date_col(df)

    big, feat_cols, date_col = build_all_players_dataset(df, window=args.window)

    X = big[feat_cols].to_numpy(dtype=np.float32)
    y = big["y_log"].to_numpy(dtype=np.float32)

    X_train, y_train, X_val, y_val, X_test, y_test = time_split(X, y)

    print(f"Training rows: {len(X)} | Features: {len(feat_cols)} | Window: {args.window}")
    model, scaler = train_regression_model(X_train, y_train, X_val, y_val)

    # Test MAE in real points (convert back)
    X_test_s = scaler.transform(X_test)
    with torch.no_grad():
        pred_log = model(torch.tensor(X_test_s, dtype=torch.float32)).squeeze(1).numpy()
    pred_pts = np.expm1(pred_log)
    true_pts = np.expm1(y_test)
    test_mae_pts = float(np.mean(np.abs(pred_pts - true_pts)))
    print(f"\nTest MAE (points): {test_mae_pts:.3f}")

    # Baseline: last-N average for that player
    one_points = pd.to_numeric(df[df["personId"] == args.person_id]["points"], errors="coerce").dropna()
    if len(one_points) >= args.window:
        avg_last = float(one_points.tail(args.window).mean())
        print(f"Last {args.window}-game avg (points): {avg_last:.1f}")

    next_pts = predict_next_for_player(df, args.person_id, feat_cols, date_col, args.window, model, scaler)
    print(f"\nPredicted NEXT game points for personId={args.person_id}: {next_pts:.1f}")

if __name__ == "__main__":
    main()

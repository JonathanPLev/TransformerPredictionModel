# src/mini_nn/lalit_model.py
#
# Mini neural net baseline on PlayerStatistics.csv
# IMPORTANT: avoids target leakage by dropping box-score scoring columns
# when the target is derived from points.

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader


# ---- training config ----
EPOCHS = 50
PATIENCE = 8
LEARNING_RATE = 5e-4
BATCH_SIZE = 256
SEED = 42


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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_player_stats_csv() -> pd.DataFrame:
    # file: src/nba_dataset/Data/PlayerStatistics.csv
    current_dir = Path(__file__).resolve().parent  # .../src/mini_nn
    data_path = current_dir.parent / "nba_dataset" / "Data" / "PlayerStatistics.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find: {data_path}")

    return pd.read_csv(data_path, low_memory=False)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = load_player_stats_csv()

    # ---- define a target (binary classification) ----
    # Example: did the player score 20+ points in the game?
    if "points" not in df.columns:
        raise ValueError("Column 'points' not found in PlayerStatistics.csv")

    points = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    y = (points >= 20).astype(np.float32)

    # ---- build features ----
    X = df.copy()

    # IDs / text / labels that shouldn't be features
    id_text_cols = [
        "firstName",
        "lastName",
        "personId",
        "gameId",
        "gameDateTimeEst",
        "playerteamCity",
        "playerteamName",
        "opponentteamCity",
        "opponentteamName",
        "gameLabel",
        "gameSubLabel",
    ]

    # Target + leakage columns:
    # These directly determine points (or are derived from it),
    # so keeping them makes metrics unrealistically perfect.
    leakage_cols = [
        "points",
        "fieldGoalsMade",
        "threePointersMade",
        "freeThrowsMade",
        "fieldGoalsAttempted",
        "threePointersAttempted",
        "freeThrowsAttempted",
        "fieldGoalsPercentage",
        "threePointersPercentage",
        "freeThrowsPercentage",
        # NOTE: "numMinutes" is still post-game, but not a direct function of points.
        # If you want stricter "pre-game only", you should engineer rolling averages
        # from prior games and drop any same-game box-score stats.
    ]

    X = X.drop(columns=id_text_cols + leakage_cols, errors="ignore")

    # Keep numeric only
    X = (
        X.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after filtering. Inspect the CSV columns.")

    print(f"Rows:     {len(df):,}")
    print(f"Features: {X.shape[1]}")
    print(f"Positive rate (>=20 pts): {y.mean():.3f}")
    print(f"Using features: {list(X.columns)}")

    # ---- split (stratified) ----
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=SEED, stratify=y_temp
    )  # ~15% val

    # ---- scale ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ---- dataloaders ----
    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s, dtype=torch.float32),
        torch.tensor(np.array(y_val), dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_s, dtype=torch.float32),
        torch.tensor(np.array(y_test), dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- model ----
    model = MiniNet(input_size=X_train.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    # Handle class imbalance a bit (>=20 is ~13% positives)
    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    if pos > 0:
        pos_weight = torch.tensor([neg / pos], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    best_val_f1 = -1.0
    best_state = None
    patience = 0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # ---- validation ----
        model.eval()
        probs_list, preds_list, labels_list = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                probs_list.append(probs.cpu().numpy())
                preds_list.append(preds.cpu().numpy())
                labels_list.append(yb.cpu().numpy())

        probs = np.concatenate(probs_list).ravel()
        preds = np.concatenate(preds_list).ravel()
        labels = np.concatenate(labels_list).ravel()

        val_f1 = f1_score(labels, preds, zero_division=0)
        val_acc = (preds == labels).mean()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"train_loss {train_loss/len(train_loader):.4f} | "
                f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}"
            )

        if patience >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"(best epoch {best_epoch}, best val_f1 {best_val_f1:.4f})"
            )
            break

    # ---- test ----
    model.load_state_dict(best_state)
    model.eval()

    probs_list, preds_list, labels_list = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            labels_list.append(yb.cpu().numpy())

    probs = np.concatenate(probs_list).ravel()
    preds = np.concatenate(preds_list).ravel()
    labels = np.concatenate(labels_list).ravel()

    test_acc = (preds == labels).mean()
    test_f1 = f1_score(labels, preds, zero_division=0)
    test_auc = roc_auc_score(labels, probs) if len(np.unique(labels)) == 2 else float("nan")

    print("\n=== FINAL TEST ===")
    print(f"Best epoch:    {best_epoch}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1:       {test_f1:.4f}")
    print(f"Test AUC:      {test_auc:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from mini_nn.models.arnav_mini_nn import SimpleNet

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 256
HIDDEN = 64

def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "Data" / "PlayerStatistics.csv"
    df = pd.read_csv(data_path, low_memory=False)

    target_col = "win"
    if target_col not in df.columns:
        raise ValueError("Expected column 'win' in Data/PlayerStatistics.csv")

    # ensure binary 0/1
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    y = (y > 0).astype(int)

    drop_candidates = {
        target_col,
        "firstName", "lastName",
        "personId", "gameId",
        "gameDate", "gameType",
        "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9",
        "Unnamed: 11", "Unnamed: 12",
    }

    X = df.drop(columns=[c for c in drop_candidates if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"Loaded {data_path} with shape {df.shape}")
    print(f"Rows used: {len(X)} | Features detected: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_test_t = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleNet(input_size=X_train_t.shape[1], hidden_size=HIDDEN, output_size=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    probs_list, preds_list, y_list = [], [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds = (probs > 0.5).astype(int)
            probs_list.append(probs)
            preds_list.append(preds)
            y_list.append(yb.cpu().numpy().reshape(-1).astype(int))

    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    y_true = np.concatenate(y_list)

    acc = (preds == y_true).mean()
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)

    print("\nTest Metrics (predict win)")
    print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"F1:       {f1:.4f}")
    print(f"AUC:      {auc:.4f}")

if __name__ == "__main__":
    main()

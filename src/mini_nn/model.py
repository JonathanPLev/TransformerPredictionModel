import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RANGE = 100


class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden1=128, hidden2=64, hidden_3=32, output_size=1):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden2, hidden_3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_3, output_size),
        )

    def forward(self, x):
        return self.nn(x)


criterion = nn.BCEWithLogitsLoss()  # binary classification

# Resolve data/model paths so the script can be run from any CWD.
MODULE_ROOT = Path(__file__).resolve().parent  # src/mini_nn
DATA_DIR = MODULE_ROOT.parent / "data_generation"
MODELS_DIR = MODULE_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
dataset_path = DATA_DIR / "NBA_Multi_Player_Training_Data.csv"
mapping_path = DATA_DIR / "player_id_mapping.csv"

df = pd.read_csv(dataset_path)
current_game_stats = [
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "PLUS_MINUS",
    "VIDEO_AVAILABLE",
    "Player_ID",
]

X = df.drop(
    ["Beats_Projected_Line", "GAME_DATE", "Player_Name"] + current_game_stats,
    axis=1,
    errors="ignore",
)
X = X.select_dtypes(include=[np.number])
y = df["Beats_Projected_Line"]

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1)

input_size = X_train.shape[1]

model = PredictionModel(input_size=input_size, hidden1=128, hidden2=64, hidden_3=32)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

best_test_loss = float("inf")
patience = 10
patience_counter = 0
best_model_state = None

for epoch in range(RANGE):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = criterion(test_pred, y_test_tensor)
        test_probs = torch.sigmoid(test_pred)
        test_predictions = (test_probs > 0.5).float()
        accuracy = (test_predictions == y_test_tensor).float().mean()

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1

        with torch.no_grad():
            test_probs = torch.sigmoid(test_pred)
            test_predictions = (test_probs > 0.5).float()
            best_accuracy = (test_predictions == y_test_tensor).float().mean().item()
    else:
        patience_counter += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{RANGE}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}"
        )

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch + 1}")
        print(f"Best test loss: {best_test_loss:.4f} at epoch {best_epoch}")
        print(f"Test Accuracy at best model: {best_accuracy:.4f}")

        model.load_state_dict(best_model_state)

        model_save_path = MODELS_DIR / "best_model.pth"
        save_dict = {
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_loss": best_test_loss.item(),
            "test_accuracy": best_accuracy,
            "input_size": input_size,
            "hidden1": (
                model.hidden1.out_features if hasattr(model, "hidden1") else None
            ),
            "hidden2": (
                model.hidden2.out_features if hasattr(model, "hidden2") else None
            ),
        }
        torch.save(save_dict, model_save_path)
        print(f"Model saved to {model_save_path}")
        break

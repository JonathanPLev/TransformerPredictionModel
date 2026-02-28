import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

# =========================
# CONFIG
# =========================
EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-4  # lower to prevent NaNs
BATCH_SIZE = 64
SAMPLE_SIZE = 100000  # reduce dataset for testing speed

# =========================
# MODEL
# =========================
class PredictorModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# DATA LOADING
# =========================
def load_data_from_csv():
    CURRENT_DIR = Path(__file__).resolve().parent
    DATA_DIR = CURRENT_DIR.parent.parent / "Data"  # points to TransformerPredictionModel/Data/
    player_file = DATA_DIR / "PlayerStatistics.csv"
    if not player_file.exists():
        raise FileNotFoundError(f"{player_file} does not exist. Make sure Data/ contains PlayerStatistics.csv")

    # read CSV safely
    df = pd.read_csv(player_file, low_memory=False)
    df.columns = df.columns.str.strip()  # remove whitespace
    return df

# =========================
# TRAINING PIPELINE
# =========================
def main():
    print("Loading data from CSVs...")
    df = load_data_from_csv()
    print(f"Total rows: {len(df)}")

    # -------------------------
    # Reduce dataset size for fast testing
    # -------------------------
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)

    # -------------------------
    # Create binary target
    # -------------------------
    df['beats_line'] = (df['points'] >= 20).astype(int)

    # -------------------------
    # Numeric features
    # -------------------------
    drop_cols = ['beats_line', 'firstName', 'lastName', 'personId', 'gameId', 'gameDateTimeEst',
                 'playerteamCity','playerteamName','opponentteamCity','opponentteamName',
                 'gameType','gameLabel','gameSubLabel','seriesGameNumber','win','home']

    X = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')

    # Handle NaNs / infinities
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    y = df['beats_line']

    print(f"Features: {X.shape[1]}")
    print("Label distribution:")
    print(y.value_counts(normalize=True))

    # -------------------------
    # Class imbalance weighting
    # -------------------------
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]], dtype=torch.float)
    print(f"Class weight for positive class: {pos_weight.item():.2f}")

    # -------------------------
    # SPLIT
    # -------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create DataLoaders
    train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values).unsqueeze(1))
    val_ds = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val.values).unsqueeze(1))
    test_ds = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test.values).unsqueeze(1))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # -------------------------
    # MODEL SETUP
    # -------------------------
    model = PredictorModel(input_size=X_train_scaled.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1 = 0
    patience_counter = 0
    best_state = None
    best_epoch = 0

    print("\nTraining...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            if torch.isnan(loss):
                print("NaN loss detected, stopping...")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                all_preds.extend(preds.numpy())
                all_probs.extend(probs.numpy())
                all_labels.extend(yb.numpy())

        all_preds = np.array(all_preds).flatten()
        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()

        val_f1 = f1_score(all_labels, all_preds)

        # Early stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = model.state_dict()
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    if best_state is None:
        print("No valid model trained. Exiting...")
        return

    # -------------------------
    # TEST
    # -------------------------
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_probs, test_labels = [], [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            test_preds.extend(preds.numpy())
            test_probs.extend(probs.numpy())
            test_labels.extend(yb.numpy())

    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_labels = np.array(test_labels).flatten()

    print("\n================ RESULTS ================")
    print(f"Best Epoch:        {best_epoch}")
    print(f"Test Accuracy:     {(test_preds == test_labels).mean():.4f}")
    print(f"Test F1 Score:     {f1_score(test_labels, test_preds):.4f}")
    print(f"Test ROC-AUC:      {roc_auc_score(test_labels, test_probs):.4f}")

    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    out = pd.DataFrame({
        "prob_beats_20_pts": test_probs,
        "prediction": test_preds,
        "actual": test_labels
    })

    out_path = Path(__file__).parent / "model_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions → {out_path}")

if __name__ == "__main__":
    main()

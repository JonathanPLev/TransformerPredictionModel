import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path


EPOCHS = 150            
PATIENCE = 15             
LEARNING_RATE = 0.0005    
BATCH_SIZE = 64

class PredictorModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1)
        )
        
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.output(x)


def main():
    CURRENT_DIR = Path(__file__).resolve().parent
    DATA_DIR = CURRENT_DIR.parent / "data_generation"
    dataset_path = DATA_DIR / "NBA_Multi_Player_Training_Data.csv"

    df = pd.read_csv(dataset_path)
    
    current_game_stats = [
        "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
        "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
        "VIDEO_AVAILABLE", "Player_ID"
    ]
    
    X = df.drop(
        ["Beats_Projected_Line", "GAME_DATE", "Player_Name"] + current_game_stats,
        axis=1,
        errors="ignore",
    )
    
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df["Beats_Projected_Line"]
    
    print(f"Features detected: {X.shape[1]}")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val.values).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test.values).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PredictorModel(input_size=X_train.shape[1])
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()


    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_pred = model(X_batch)
                val_loss += criterion(val_pred, y_batch).item()
                
                val_probs = torch.sigmoid(val_pred)
                predictions = (val_probs > 0.5).float()
                
                all_preds.extend(predictions.numpy())
                all_probs.extend(val_probs.numpy())
                all_labels.extend(y_batch.numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()
        
        val_accuracy = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds)
        

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            test_pred = model(X_batch)
            test_prob = torch.sigmoid(test_pred)
            predictions = (test_prob > 0.5).float()
            
            test_preds.extend(predictions.numpy())
            test_probs.extend(test_prob.numpy())
            test_labels.extend(y_batch.numpy())
    
    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_labels = np.array(test_labels).flatten()
    
    test_accuracy = (test_preds == test_labels).mean()
    test_f1 = f1_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)

    print(f"Best Model found at Epoch: {best_epoch}")
    print(f"Final Test Accuracy:       {test_accuracy:.4f} ({(test_accuracy*100):.1f}%)")
    print(f"Final Test F1 Score:       {test_f1:.4f}")
    print(f"Final Test AUC-ROC:        {test_auc:.4f}")
    


if __name__ == "__main__":
    main()
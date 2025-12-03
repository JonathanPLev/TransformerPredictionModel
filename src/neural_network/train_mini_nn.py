
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_network.mini_nn import MiniNN


class MiniNNTrainer:
    
    def __init__(self, data_path, target="Beats_Projected_Line"):
        
        self.data_path = data_path
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
    
    def prepare_data(self):
        
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        X = df.drop([self.target, "GAME_DATE"], axis=1, errors='ignore')
        
        current_game_stats = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE",
            "Player_Name", "PLAYER_ID"
        ]
        
        cols_to_drop = [col for col in current_game_stats if col in X.columns]
        X = X.drop(cols_to_drop, axis=1, errors='ignore')
        
        X = X.select_dtypes(include=[np.number])
        
        X = X.fillna(X.mean())
        
        y = df[self.target].values
        
        self.feature_columns = X.columns.tolist()
        
        print("\nData Summary:")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Samples: {len(X)}")
        print("Target distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            label = "Over" if cls == 1 else "Under"
            print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
        
        return X.values, y
    
    def train(self, epochs=50, batch_size=32, lr=0.001, hidden_dim=64):
        
        X, y = self.prepare_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = MiniNN(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=hidden_dim
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print("\nTraining Mini Neural Network...")
        print(f"Architecture: {X_train_scaled.shape[1]} -> {hidden_dim} -> 32 -> 2")
        print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}\n")
        
        train_losses = []
        train_accs = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            train_losses.append(avg_loss)
            train_accs.append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                      f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
        
        self.model.eval()
        with torch.no_grad():
            X_test_device = X_test_tensor.to(self.device)
            outputs = self.model(X_test_device)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.cpu().numpy()
        
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*50}")
        print("TRAINING COMPLETE")
        print(f"{'='*50}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Under', 'Over']))
        
        return {
            'test_accuracy': test_accuracy,
            'train_losses': train_losses,
            'train_accs': train_accs
        }
    
    def save_model(self, model_name="mini_nn"):
        
        if self.model is None:
            print("No model to save. Train first.")
            return
        
        os.makedirs('models', exist_ok=True)
        
        model_path = f"models/{model_name}_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }, model_path)
        
        print(f"\nModel saved: {model_path}")


def main():
    data_path = "src/data_generation/NBA_Multi_Player_Training_Data.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run the data generation pipeline first:")
        print("  cd src/data_generation")
        print("  python nba_pipeline_data.py")
        return
    
    trainer = MiniNNTrainer(data_path)
    trainer.train(epochs=50, batch_size=32, lr=0.001, hidden_dim=64)
    trainer.save_model()


if __name__ == "__main__":
    main()
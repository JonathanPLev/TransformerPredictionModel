
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt

#define model

class MiniNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


# training

class MiniNNTrainer:
    
    def __init__(self, data_path, target="Beats_Projected_Line"):
        self.data_path = data_path
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
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
    
    def train(self, epochs=50, batch_size=32, lr=0.001, hidden_dim=64, val_split=0.2):
        
        X, y = self.prepare_data()
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model = MiniNN(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=hidden_dim
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\n{'='*60}")
        print("Training Mini Neural Network")
        print(f"{'='*60}")
        print(f"Architecture: {X_train_scaled.shape[1]} -> {hidden_dim} -> 32 -> 2")
        print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Training phase
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
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                X_val_device = X_val_tensor.to(self.device)
                y_val_device = y_val_tensor.to(self.device)
                
                outputs = self.model(X_val_device)
                loss = criterion(outputs, y_val_device)
                
                val_loss = loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total = y_val_device.size(0)
                val_correct = (predicted == y_val_device).sum().item()
            
            val_acc = val_correct / val_total
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1:3d}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        self.model.eval()
        with torch.no_grad():
            X_val_device = X_val_tensor.to(self.device)
            outputs = self.model(X_val_device)
            _, y_pred = torch.max(outputs.data, 1)
            y_pred = y_pred.cpu().numpy()
        
        final_accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final Validation Accuracy: {final_accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Under', 'Over']))
        
        cm = confusion_matrix(y_val, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'final_accuracy': final_accuracy,
            'history': self.history,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_val, y_pred, 
                                                          target_names=['Under', 'Over'],
                                                          output_dict=True)
        }
    
    def plot_results(self, save_dir='src/mini_nn/models'):
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'{save_dir}/training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining plots saved to: {plot_path}")
        plt.close()
    
    def save_model(self, save_path='src/mini_nn/models/nidhi_mini_nn.pt'):
        if self.model is None:
            print("No model to save. Train first.")
            return
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': len(self.feature_columns),
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'history': self.history
        }, save_path)
        
        print(f"\nModel saved to: {save_path}")


# main

if __name__ == "__main__":
    data_path = "src/data_generation/NBA_Multi_Player_Training_Data.csv"
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Please run the data generation pipeline first")
        exit(1)
    
    trainer = MiniNNTrainer(data_path)
    
    #train
    results = trainer.train(
        epochs=100,
        batch_size=32,
        lr=0.001,
        hidden_dim=64
    )
    
    #plots
    trainer.plot_results()
    
    #model
    trainer.save_model()
    
    print("\n" + "="*60)
    print("ALL DONE!")
    print("="*60)
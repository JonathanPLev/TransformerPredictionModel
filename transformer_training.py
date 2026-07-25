import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from transformer_training_data import DataPreparer, PlayerTrainingData
from transformer_architecture import PlayerPerformanceTransformer


class ModelTrainer():
    def __init__(self, batch_size=64, epochs=50, device=None):

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.data = DataPreparer(batch_size=batch_size)
        self.train_loader, self.val_loader, self.test_loader = self.data.create_subsplit_loaders()

        max_home_team = self.data.df['hometeamid'].max()
        max_away_team = self.data.df['awayteamid'].max()
        num_teams = int(max(max_home_team, max_away_team) + 1)

        num_features = len(self.data.feature_cols)
        self.model = PlayerPerformanceTransformer(num_cont_features=num_features,
                num_teams=num_teams
        ).to(self.device)
        
        
    def train_model(self):
        pass


        


# def train_model():
#     # Settings
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     batch_size = 1024  # Large batch size for 1M rows
#     epochs = 50
    
#     # Load Data
#     dataset = NBADataset()
#     train_size = int(0.9 * len(dataset))
#     val_size = len(dataset) - train_size
#     train_db, val_db = torch.utils.data.random_split(dataset, [train_size, val_size])
    
#     train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)

#     # Initialize Model
#     model = PlayerPerformanceTransformer(
#         num_cont_features=dataset.x_cont.shape[2],
#         num_teams=int(dataset.x_id.max() + 1)
#     ).to(device)

#     # Optimizer & Loss
#     optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
#     criterion = nn.HuberLoss() # More robust to NBA outliers than MSE
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

#     print(f"Starting training on {device}...")
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0
#         for x_cont, x_id, y in train_loader:
#             x_cont, x_id, y = x_cont.to(device), x_id.to(device), y.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(x_cont, x_id)
#             loss = criterion(outputs, y)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         # Validation
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for x_cont, x_id, y in val_loader:
#                 x_cont, x_id, y = x_cont.to(device), x_id.to(device), y.to(device)
#                 outputs = model(x_cont, x_id)
#                 val_loss += criterion(outputs, y).item()
        
#         avg_val = val_loss/len(val_loader)
#         scheduler.step(avg_val)
#         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

#     # Save the weights
#     torch.save(model.state_dict(), 'nba_transformer_weights.pth')
#     print("Training complete.")

# if __name__ == "__main__":
#     train_model()
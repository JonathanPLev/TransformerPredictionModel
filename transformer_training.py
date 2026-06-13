import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PlayerPerformanceTransformer(nn.Module):
    def __init__(self, num_cont_features, num_teams, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. Continuous Stream: Projects the ~50 engineered stats
        self.cont_projection = nn.Linear(num_cont_features, d_model)
        
        # 2. Categorical Stream: Learns "team identity" (Celtics vs. Pistons)
        # We split d_model between Player Team and Opponent Team
        self.team_embedding = nn.Embedding(num_teams, d_model // 2)
        
        # 3. Positional Encoding: Tells the model which game is "Game 1" vs "Game 10"
        self.pos_embedding = nn.Parameter(torch.zeros(1, 10, d_model))
        
        # 4. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Head: Predicts [Points, Assists, Rebounds, Minutes]
        self.output_head = nn.Linear(d_model, 4)

    def forward(self, x_cont, x_id):
        # x_id shape: (batch, 10, 2)
        # Embed Player Team and Opponent Team
        p_team_emb = self.team_embedding(x_id[:, :, 0])
        o_team_emb = self.team_embedding(x_id[:, :, 1])
        
        # Combine embeddings into a single d_model vector
        id_emb = torch.cat([p_team_emb, o_team_emb], dim=-1) 
        
        # Project continuous stats and fuse with IDs
        x = self.cont_projection(x_cont) + id_emb + self.pos_embedding
        
        # Pass through Transformer blocks
        # For sequence-to-one, we take the representation of the last game in the window
        x = self.transformer(x)
        last_token = x[:, -1, :] 
        
        return self.output_head(last_token)
        
class NBADataset(Dataset):
    def __init__(self, prefix='nba_full_data'):
        self.x_cont = torch.from_numpy(np.load(f'{prefix}_X_continuous.npy')).float()
        self.x_id = torch.from_numpy(np.load(f'{prefix}_X_ids.npy')).long()
        self.y = torch.from_numpy(np.load(f'{prefix}_y_targets.npy')).float()

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x_cont[idx], self.x_id[idx], self.y[idx]

def train_model():
    # Settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 1024  # Large batch size for 1M rows
    epochs = 50
    
    # Load Data
    dataset = NBADataset()
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_db, val_db = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)

    # Initialize Model
    model = PlayerPerformanceTransformer(
        num_cont_features=dataset.x_cont.shape[2],
        num_teams=int(dataset.x_id.max() + 1)
    ).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.HuberLoss() # More robust to NBA outliers than MSE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_cont, x_id, y in train_loader:
            x_cont, x_id, y = x_cont.to(device), x_id.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_cont, x_id)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_cont, x_id, y in val_loader:
                x_cont, x_id, y = x_cont.to(device), x_id.to(device), y.to(device)
                outputs = model(x_cont, x_id)
                val_loss += criterion(outputs, y).item()
        
        avg_val = val_loss/len(val_loader)
        scheduler.step(avg_val)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

    # Save the weights
    torch.save(model.state_dict(), 'nba_transformer_weights.pth')
    print("Training complete.")

if __name__ == "__main__":
    train_model()
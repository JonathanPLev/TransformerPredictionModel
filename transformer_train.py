import torch
import torch.nn as nn
import torch.optim as optim

from transformer_training_data import DataPreparer
from transformer_arch import PlayerPerformanceTransformer


class ModelTrainer():
    def __init__(self, batch_size=64, epochs=50, device=None):

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.data = DataPreparer(batch_size=batch_size)
        self.train_loader, self.val_loader, self.test_loader = self.data.create_subsplit_loaders()

        max_home_team = self.data.df['player_team_idx'].max()
        max_away_team = self.data.df['opp_team_idx'].max()
        self.num_teams = int(max(max_home_team, max_away_team) + 1)

        self.num_features = len(self.data.feature_cols)
        self.model = PlayerPerformanceTransformer(num_cont_features=self.num_features,
                num_teams=self.num_teams
        ).to(self.device)
        
        
    def train_model(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=0.01)
        criterion = nn.HuberLoss()

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for x_cont, x_id, y in self.train_loader:
                x_cont, x_id, y = x_cont.to(self.device), x_id.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x_cont, x_id)
                l1_loss_fn = nn.L1Loss(reduction='none')
                raw_losses_per_batch = l1_loss_fn(outputs, y)

                # 2. Average across the batch to get a shape of [4]
                raw_losses = raw_losses_per_batch.mean(dim=0) 

                # 3. Apply Homoscedastic Uncertainty Weighting
                log_vars = torch.clamp(self.model.log_vars, min=-5.0, max=5.0)
                weighted_losses = (torch.exp(-log_vars) * raw_losses) + log_vars
                # 4. Sum to get the final scalar loss and backpropagate
                total_loss = torch.sum(weighted_losses)
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += total_loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            total_val_samples = 0
            val_mae_sum = torch.zeros(4, device=self.device)
            with torch.no_grad():
                for x_cont, x_id, y in self.val_loader:
                    x_cont, x_id, y = x_cont.to(self.device), x_id.to(self.device), y.to(self.device)
                    outputs = self.model(x_cont, x_id)
                    val_loss += criterion(outputs, y).item()
                    mae_per_target = torch.abs(outputs - y).sum(dim=0) # shape: (4,)
                    val_mae_sum += mae_per_target
                    total_val_samples += y.size(0)
            
            avg_val = val_loss/len(self.val_loader)
            avg_mae = val_mae_sum / total_val_samples
            scheduler.step(avg_val)
            print(f"Epoch {epoch+1:02d}/{self.epochs} | Train Loss: {train_loss/len(self.train_loader):.4f} | Val Loss: {avg_val:.4f}")
            # print(f"-> Pts MAE: {avg_mae[0]:.2f} | Ast MAE: {avg_mae[1]:.2f} | Reb MAE: {avg_mae[2]:.2f} | Min MAE: {avg_mae[3]:.2f}")
            print(f"-> Pts MAE: {avg_mae[0]:.2f} | Mins MAE: {avg_mae[1]:.2f} | Reb MAE: {avg_mae[2]:.2f} | Ast MAE: {avg_mae[3]:.2f}")

        # Save the weights
        torch.save(self.model.state_dict(), 'nba_transformer_weights.pth')
        print("Training complete.")

    def test_model(self):
        model = PlayerPerformanceTransformer(num_cont_features=self.num_features,
                num_teams=self.num_teams).to(self.device)
        state_dict = torch.load('nba_transformer_weights.pth', weights_only=True)
        model.load_state_dict(state_dict)
        _,_, test_loader = self.data.create_subsplit_loaders()
        model.eval()
        total_val_samples = 0
        val_mae_sum = torch.zeros(4, device=self.device)
        with torch.no_grad():
            for x_cont, x_id, y in test_loader:
                x_cont, x_id, y = x_cont.to(self.device), x_id.to(self.device), y.to(self.device)
                outputs = model(x_cont, x_id)
                mae_per_target = torch.abs(outputs - y).sum(dim=0) # shape: (4,)
                val_mae_sum += mae_per_target
                total_val_samples += y.size(0)
        avg_mae = val_mae_sum / total_val_samples
        print(f"-> Pts MAE: {avg_mae[0]:.2f} | Mins MAE: {avg_mae[1]:.2f} | Reb MAE: {avg_mae[2]:.2f} | Ast MAE: {avg_mae[3]:.2f}")




if __name__ == "__main__":
    modeltrainer = ModelTrainer()
    # modeltrainer.train_model()
    modeltrainer.test_model()



"""
NBA Player Transformer — Training Script

Loads train.pt / val.pt produced by data_pipeline.py and trains the
NBAPlayerTransformer from model.py.

Usage:
    python train.py
    python train.py --epochs 100 --batch-size 512 --lr 5e-4

Output:
    nba_transformer_v2.pth  – best model weights (by val loss)
"""

import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import NBAPlayerTransformer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NBASequenceDataset(Dataset):
    def __init__(self, path: str):
        data = torch.load(path, weights_only=False)
        self.x_cont = data["x_cont"]
        self.x_person_id = data["x_person_id"]
        self.x_player_team = data["x_player_team"]
        self.x_opp_team = data["x_opp_team"]
        self.x_game_idx = data["x_game_idx"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x_cont[idx],
            self.x_person_id[idx],
            self.x_player_team[idx],
            self.x_opp_team[idx],
            self.x_game_idx[idx],
            self.y[idx],
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    d_model: int = 64,
    nhead: int = 8,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.1,
    output_path: str = "nba_transformer_v2.pth",
    meta_path: str = "pipeline_meta.pkl",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata for model dimensions
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    num_cont = meta["num_cont_features"]
    num_players = meta["num_players"]
    num_teams = meta["num_teams"]
    seq_len = meta["seq_len"]
    print(f"  num_cont={num_cont}, num_players={num_players}, num_teams={num_teams}, seq_len={seq_len}")

    # Datasets & loaders
    train_ds = NBASequenceDataset("train.pt")
    val_ds = NBASequenceDataset("val.pt")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = NBAPlayerTransformer(
        num_cont_features=num_cont,
        num_players=num_players,
        num_teams=num_teams,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        seq_len=seq_len,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    best_val_loss = float("inf")

    print(f"\nStarting training for {epochs} epochs...\n")
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for x_cont, x_pid, x_pt, x_ot, x_gi, y in train_loader:
            x_cont = x_cont.to(device)
            x_pid  = x_pid.to(device)
            x_pt   = x_pt.to(device)
            x_ot   = x_ot.to(device)
            x_gi   = x_gi.to(device)
            y      = y.to(device)

            optimizer.zero_grad()
            preds = model(x_cont, x_pid, x_pt, x_ot, x_gi)
            loss = weighted_huber_loss(preds, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        total_pts_mse = 0.0
        num_val_samples = 0.0
        with torch.no_grad():
            for x_cont, x_pid, x_pt, x_ot, x_gi, y in val_loader:
                x_cont = x_cont.to(device)
                x_pid  = x_pid.to(device)
                x_pt   = x_pt.to(device)
                x_ot   = x_ot.to(device)
                x_gi   = x_gi.to(device)
                y      = y.to(device)
                preds = model(x_cont, x_pid, x_pt, x_ot, x_gi)
                val_loss += weighted_huber_loss(preds, y).item()
                pts_diff = (preds[:,0] - y[:,0]) ** 2
                total_pts_mse += pts_diff.sum().item()
                num_val_samples += y.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        pts_rmse = (total_pts_mse / num_val_samples) ** 0.5
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Points RMSE: {pts_rmse:.2f}"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ New best val loss: {best_val_loss:.4f} — saved to {output_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {output_path}")


def weighted_huber_loss(preds, target):
    # Higher weights for stats with lower natural ranges (Assists/Rebounds)
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=preds.device)
    
    # Calculate Huber for each stat
    loss_per_stat = torch.nn.functional.huber_loss(preds, target, reduction='none', delta=1.5)
    
    # Apply weights and average
    weighted_loss = loss_per_stat * weights
    return weighted_loss.mean()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--output", type=str, default="nba_transformer_v2.pth")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        output_path=args.output,
    )

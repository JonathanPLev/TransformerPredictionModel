"""
Decoder-only GPT-style Transformer for NBA player performance prediction.

Predicts [points, assists, rebounds, minutes] for the next game given a
K=10 game sliding window as input.
"""

import torch
import torch.nn as nn


class NBAPlayerTransformer(nn.Module):
    """
    Decoder-only transformer that predicts 4 next-game stats from a K-length
    sequence of past-game feature vectors.

    Inputs per forward call:
        x_cont      (batch, K, num_cont)   – continuous features
        x_person_id (batch,)               – player ID (same for all timesteps)
        x_player_team (batch, K)           – player team ID per timestep
        x_opp_team  (batch, K)             – opponent team ID per timestep
        x_game_idx  (batch, K)             – season game index (1–82) per timestep

    Output:
        (batch, 4)  – predicted [points, assists, rebounds, minutes]
    """

    def __init__(
        self,
        num_cont_features: int,
        num_players: int,
        num_teams: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        seq_len: int = 11,  # K past games + 1 target context token
        max_game_idx: int = 83,  # season game indices 1-82 + buffer
    ):
        super().__init__()
        self.seq_len = seq_len

        # --- Embedding layers ---
        # person_id embedding: size 16 per instructions
        self.player_embedding = nn.Embedding(num_players, 16)
        # player_team and opponent_team embeddings: size 8 each
        self.player_team_embedding = nn.Embedding(num_teams, 8)
        self.opp_team_embedding = nn.Embedding(num_teams, 8)

        # Total input dim after concatenation: num_cont + 16 + 8 + 8
        input_dim = num_cont_features + 16 + 8 + 8
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional embedding indexed by season game index (1-82)
        self.pos_embedding = nn.Embedding(max_game_idx, d_model)

        # Decoder-only transformer: TransformerEncoder + causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.final_norm = nn.LayerNorm(d_model)
        
        # Regression head → [points, assists, rebounds, minutes]
        self.output_head = nn.Linear(d_model, 4)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.zeros_(self.output_head.bias)

    def forward(
        self,
        x_cont: torch.Tensor,
        x_person_id: torch.Tensor,
        x_player_team: torch.Tensor,
        x_opp_team: torch.Tensor,
        x_game_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, K, _ = x_cont.shape

        # Embed categorical features
        # person_id is per-player (batch,) → expand to (batch, K, 16)
        player_emb = self.player_embedding(x_person_id).unsqueeze(1).expand(B, K, -1)
        team_emb = self.player_team_embedding(x_player_team)   # (B, K, 8)
        opp_emb = self.opp_team_embedding(x_opp_team)          # (B, K, 8)

        # Concatenate all inputs and project to d_model
        x = torch.cat([x_cont, player_emb, team_emb, opp_emb], dim=-1)  # (B, K, input_dim)
        x = self.input_proj(x)  # (B, K, d_model)

        # Add learned positional encoding by season game index
        pos = self.pos_embedding(x_game_idx)  # (B, K, d_model)
        x = x + pos

        # Causal (decoder-only) mask: each position can only attend to itself
        # and earlier positions
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            K, device=x.device, dtype=x.dtype
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Sequence-to-one: take last token representation
        last = x[:, -1, :]  # (B, d_model)
        last = self.final_norm(last)
        return self.output_head(last)  # (B, 4)

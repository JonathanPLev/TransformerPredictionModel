import torch
import torch.nn as nn

class PlayerPerformanceTransformer(nn.Module):
    def __init__(self, num_cont_features, num_teams, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        
        # 1. Continuous Stream: Projects the ~50 engineered stats
        self.cont_projection = nn.Sequential(
            nn.Linear(num_cont_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # 2. Categorical Stream: Learns "team identity" (Celtics vs. Pistons)
        # We split d_model between Player Team and Opponent Team
        self.team_embedding = nn.Embedding(num_teams, d_model // 2)
        
        # 3. Positional Encoding: Tells the model which game is "Game 1" vs "Game 10"
        self.pos_embedding = nn.Parameter(torch.zeros(1, 11, d_model))
        
        # 4. Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 5. Output Head: Predicts [Points, Assists, Rebounds, Minutes]
        self.output_head = nn.Linear(d_model, 4)

        self.log_vars = nn.Parameter(torch.zeros(4))

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
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        last_token = x[:, -1, :] 
        
        return self.output_head(last_token)
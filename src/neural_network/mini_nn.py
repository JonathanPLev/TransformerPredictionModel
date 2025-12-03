import torch.nn as nn

class MiniNN(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.model(x)
        return self.softmax(logits)

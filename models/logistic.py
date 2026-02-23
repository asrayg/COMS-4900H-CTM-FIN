import torch
import torch.nn as nn
from .base import StreamingModel

class LogisticStream(StreamingModel):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward_step(self, x_t, state=None):
        # x_t shape: (features,) or (batch, features)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)  # add batch
        logits = self.net(x_t).squeeze(-1)
        return logits, None   # stateless

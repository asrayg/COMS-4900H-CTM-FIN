import torch
import torch.nn as nn
from .base import StreamingModel

class LSTMStream(StreamingModel):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def reset_state(self, batch_size=1):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return (h0, c0)
    
    def forward_step(self, x_t, state):
        # x_t: (batch, features) or (features,)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0).unsqueeze(1)  # (1, 1, F)
            batch_first = True
        else:
            x_t = x_t.unsqueeze(1)  # (batch, 1, F)
            
        # Ensure state is on same device as input
        if state is not None:
            state = (state[0].to(x_t.device), state[1].to(x_t.device))
        
        out, (h, c) = self.lstm(x_t, state)
        logits = self.fc(out[:, -1, :]).squeeze(-1)
        return logits, (h, c)

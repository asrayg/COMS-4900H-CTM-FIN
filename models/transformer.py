import torch
import torch.nn as nn
from .base import StreamingModel

class TransformerWindow(StreamingModel):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, window=40):
        super().__init__()
        self.window = window
        self.input_dim = input_dim
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(window, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def reset_state(self, batch_size=1):
        # State is the sliding window buffer: (batch, 0, input_dim)
        buffer = torch.zeros(batch_size, 0, self.input_dim)
        return buffer

    def forward(self, x_seq):
        # x_seq: (batch, window, input_dim)
        batch, w, _ = x_seq.shape
        x = self.embed(x_seq)  # (batch, w, d_model)
        positions = torch.arange(w, device=x.device).unsqueeze(0).expand(batch, -1)
        x = x + self.pos_encoder(positions)

        # Causal mask ensuring no future leakage within the window
        mask = torch.triu(torch.ones(w, w) * float('-inf'), diagonal=1).to(x.device)

        out = self.transformer(x, mask=mask)  # (batch, w, d_model)
        last_out = out[:, -1, :]  # (batch, d_model)
        logits = self.fc(last_out).squeeze(-1)
        return logits

    def forward_step(self, x_t, state):
        # x_t: (batch, features) or (features,)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)  # (1, features)

        buffer = state
        if buffer is not None:
            buffer = buffer.to(x_t.device)
        else:
            buffer = torch.zeros(x_t.size(0), 0, self.input_dim, device=x_t.device)

        # Append new timestep to buffer
        x_t_3d = x_t.unsqueeze(1)  # (batch, 1, features)
        buffer = torch.cat([buffer, x_t_3d], dim=1)

        # Keep only the last `window` steps
        if buffer.size(1) > self.window:
            buffer = buffer[:, -self.window:, :]

        # Pad if shorter than window
        current_len = buffer.size(1)
        if current_len < self.window:
            pad = torch.zeros(buffer.size(0), self.window - current_len, self.input_dim,
                              device=buffer.device)
            x_seq = torch.cat([pad, buffer], dim=1)
        else:
            x_seq = buffer

        logits = self.forward(x_seq)
        return logits, buffer

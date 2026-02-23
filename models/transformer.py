import torch
import torch.nn as nn

class TransformerWindow(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, window=40):
        super().__init__()
        self.window = window
        self.embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(window, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)
    
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

import torch
import torch.nn as nn

class TransformerBaseline(nn.Module):
    """
    Standard Sliding-Window Transformer.
    Recomputes representations from scratch at each step.
    """
    def __init__(self, feature_dim, num_heads, num_layers, window_size, out_dim, d_model=64):
        super().__init__()
        self.window_size = window_size
        self.d_model = d_model
        
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, out_dim)

    def forward(self, x_window):
        """
        x_window: (batch_size, window_size, feature_dim)
        """
        x = self.input_proj(x_window)
        x = x + self.pos_embedding
        x = self.transformer(x)
        # Take the last token representation
        x = x[:, -1, :]
        return self.fc(x)

class GRUBaseline(nn.Module):
    """
    Recurrent baseline (GRU). 
    Maintains a hidden state but follows standard episodic recurrence.
    """
    def __init__(self, feature_dim, hidden_dim, num_layers, out_dim):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_t, h_prev):
        """
        x_t: (batch_size, 1, feature_dim)
        h_prev: (num_layers, batch_size, hidden_dim)
        """
        out, h_curr = self.gru(x_t, h_prev)
        logits = self.fc(out[:, -1, :])
        return logits, h_curr

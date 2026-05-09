import torch
import torch.nn as nn
from .base import StreamingModel

class CTMInspired(StreamingModel):
    def __init__(self, input_dim, hidden_dim=64, history_len=64, num_pairs=128, max_ticks=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.num_pairs = num_pairs
        self.max_ticks = max_ticks

        # Recurrent cell
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

        # Sync MLP
        self.sync_mlp = nn.Sequential(
            nn.Linear(num_pairs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Internal tick pause embedding to replace external inputs
        self.pause_embed = nn.Parameter(torch.zeros(1, input_dim))

    def reset_state(self, batch_size=1):
        # state: (h, buffer, pointer)
        h = torch.zeros(batch_size, self.hidden_dim)
        buffer = torch.zeros(batch_size, self.history_len, self.hidden_dim)
        return (h, buffer, 0)

    def _update_buffer(self, buffer, ptr, new_h):
        buffer[:, ptr, :] = new_h
        new_ptr = (ptr + 1) % self.history_len
        return buffer, new_ptr

    def _compute_sync(self, buffer):
        B, L, D = buffer.size()

        # Sample num_pairs random indices to form pair relationships from the state history
        idx = torch.randint(0, L, (self.num_pairs, 2), device=buffer.device)

        # Gather vector representations for pairs
        s1 = buffer[:, idx[:, 0], :]
        s2 = buffer[:, idx[:, 1], :]

        # Neural Synchronization = Temporal Dot Product
        sync = (s1 * s2).sum(dim=2)  # shape (batch_size, num_pairs)
        return sync

    def forward_step(self, x_t, state, ticks=None, return_all_ticks=False):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)  # Add batch

        h, buffer, ptr = state
        h = h.to(x_t.device)
        buffer = buffer.to(x_t.device)

        # Update state continuously with the incoming event feature
        h_new = self.gru_cell(x_t, h)
        buffer, ptr = self._update_buffer(buffer, ptr, h_new)

        num_ticks = ticks if ticks is not None else self.max_ticks
        logits_list = []

        # Produce logits at internal tick 1
        sync = self._compute_sync(buffer)
        logits = self.sync_mlp(sync).squeeze(-1)
        logits_list.append(logits)

        # Extra continuous inference processing loops (internal ticks)
        for t in range(1, num_ticks):
            pause_input = self.pause_embed.expand(x_t.size(0), -1)
            h_new = self.gru_cell(pause_input, h_new)
            buffer, ptr = self._update_buffer(buffer, ptr, h_new)

            sync = self._compute_sync(buffer)
            logits = self.sync_mlp(sync).squeeze(-1)
            logits_list.append(logits)

        if return_all_ticks:
            return logits_list, (h_new, buffer, ptr)
        return logits_list[-1], (h_new, buffer, ptr)

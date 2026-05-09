import torch
import torch.nn as nn
from .base import StreamingModel


class VectorizedNeuronMLPs(nn.Module):
    """
    Batched per-neuron activation functions.
    Instead of looping over D neurons with individual MLPs, we use grouped
    linear layers that process all neurons in a single matmul.
    Each neuron gets its own weights (like D independent small MLPs) but
    the computation is fully vectorized.
    """
    def __init__(self, num_neurons, m_history, neuron_hidden=16):
        super().__init__()
        self.num_neurons = num_neurons
        # Layer 1: (D, m_history) -> (D, neuron_hidden) — independent per neuron
        self.w1 = nn.Parameter(torch.randn(num_neurons, m_history, neuron_hidden) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(num_neurons, neuron_hidden))
        # Layer 2: (D, neuron_hidden) -> (D, 1) — independent per neuron
        self.w2 = nn.Parameter(torch.randn(num_neurons, neuron_hidden, 1) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(num_neurons, 1))

    def forward(self, pre_hist):
        """
        pre_hist: (batch, num_neurons, m_history)
        returns:  (batch, num_neurons)
        """
        # Einstein notation: batch b, neuron d, input i, hidden h
        h = torch.einsum('bdi,dih->bdh', pre_hist, self.w1) + self.b1
        h = torch.relu(h)
        out = torch.einsum('bdh,dho->bdo', h, self.w2) + self.b2
        return out.squeeze(-1)  # (batch, num_neurons)


class CTMFull(StreamingModel):
    """
    Full Continuous Thought Machine following the Sakana AI architecture.

    Key components faithful to the paper:
    1. Synapse model: maps (post_activations, input) -> pre_activations
    2. Per-neuron activation functions: each neuron has its own MLP over
       its pre-activation history
    3. Neural synchronization readout: pairwise dot products of post-activations
       for a fixed set of neuron pairs
    4. Internal ticks: x_t only on tick 0, zero-masked input for ticks 1..T-1
    5. Dual loss (min-loss + max-certainty) applied over all tick outputs
    """
    def __init__(self, input_dim, hidden_dim=128, num_neurons=64,
                 m_history=8, max_ticks=10, num_sync_pairs=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons
        self.m_history = m_history
        self.max_ticks = max_ticks
        self.num_sync_pairs = num_sync_pairs

        # Synapse model: deeper network for richer pre-activation computation
        self.synapse_net = nn.Sequential(
            nn.Linear(num_neurons + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_neurons),
        )

        # Vectorized per-neuron activation functions
        self.neuron_acts = VectorizedNeuronMLPs(num_neurons, m_history, neuron_hidden=16)

        # Fixed synchronization pairs (deterministic, not random at eval time)
        # Register as buffer so they move with the model to GPU and are saved
        torch.manual_seed(42)
        pair_idx = torch.stack([
            torch.randint(0, num_neurons, (num_sync_pairs,)),
            torch.randint(0, num_neurons, (num_sync_pairs,)),
        ], dim=0)  # (2, num_sync_pairs)
        self.register_buffer('sync_pairs', pair_idx)

        # Output head from synchronization vector
        self.output_head = nn.Sequential(
            nn.Linear(num_sync_pairs, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def reset_state(self, batch_size=1):
        pre_hist = torch.zeros(batch_size, self.num_neurons, self.m_history)
        post_act = torch.zeros(batch_size, self.num_neurons)
        return (pre_hist, post_act)

    def _compute_sync(self, post_act):
        """Compute neural synchronization vector from fixed pairs."""
        s1 = post_act[:, self.sync_pairs[0]]  # (batch, num_sync_pairs)
        s2 = post_act[:, self.sync_pairs[1]]  # (batch, num_sync_pairs)
        return s1 * s2  # element-wise "temporal dot product"

    def forward_step(self, x_t, state, ticks=None, return_all_ticks=False):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)

        pre_hist, post_act = state
        pre_hist = pre_hist.to(x_t.device)
        post_act = post_act.to(x_t.device)

        num_ticks = ticks if ticks is not None else self.max_ticks
        logits_list = []

        # Zero input for internal ticks (only tick 0 sees the real observation)
        zero_input = torch.zeros_like(x_t)

        for t in range(num_ticks):
            # Tick 0: use real input; ticks 1+: zero-masked input
            current_input = x_t if t == 0 else zero_input

            # Synapse: combine current activations with input
            synapse_in = torch.cat([post_act, current_input], dim=-1)
            new_pre_act = self.synapse_net(synapse_in)  # (batch, num_neurons)

            # Shift history and append new pre-activations
            pre_hist = torch.cat([pre_hist[..., 1:], new_pre_act.unsqueeze(-1)], dim=-1)

            # Per-neuron activation functions (vectorized)
            post_act = self.neuron_acts(pre_hist)  # (batch, num_neurons)

            # Synchronization readout -> logits
            sync = self._compute_sync(post_act)
            logits = self.output_head(sync).squeeze(-1)
            logits_list.append(logits)

        if return_all_ticks:
            return logits_list, (pre_hist, post_act)
        return logits_list[-1], (pre_hist, post_act)

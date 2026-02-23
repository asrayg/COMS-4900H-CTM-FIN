import torch
import torch.nn as nn
from .base import StreamingModel

class CTMFull(StreamingModel):
    """
    (Optional/Bonus) Full Sakana Continuous Thought Machine
    This incorporates per-neuron MLPs and the min-loss max-certainty objective
    via sub-sampling synchronization tensors.
    """
    def __init__(self, input_dim, hidden_dim=64, num_neurons=64, m_history=8, max_ticks=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neurons = num_neurons
        self.m_history = m_history
        self.max_ticks = max_ticks
        
        # Synapse model: U-Net style
        # For boilerplate, keeping this a basic linear stack
        self.synapse_mlp = nn.Sequential(
            nn.Linear(num_neurons + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_neurons)
        )
        
        # Neuron-level models
        self.neuron_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m_history, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ) for _ in range(num_neurons)
        ])
        
        self.output_mlp = nn.Sequential(
            nn.Linear(32, 16), # D_out = 32 matching build guide
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def reset_state(self, batch_size=1):
        pre_hist = torch.zeros(batch_size, self.num_neurons, self.m_history)
        post_act = torch.zeros(batch_size, self.num_neurons)
        return (pre_hist, post_act)
    
    def forward_step(self, x_t, state, ticks=None):
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
            
        pre_hist, post_act = state
        pre_hist = pre_hist.to(x_t.device)
        post_act = post_act.to(x_t.device)
        
        num_ticks = ticks if ticks is not None else self.max_ticks
        
        for t in range(num_ticks):
            # Synapse processing
            # For purely internal ticks beyond 1, x_t should ideally be 0 or masked
            synapse_in = torch.cat([post_act, x_t], dim=-1)
            new_pre_act = self.synapse_mlp(synapse_in) # (batch, num_neurons)
            
            # Update history buffer
            pre_hist = torch.cat([pre_hist[..., 1:], new_pre_act.unsqueeze(-1)], dim=-1)
            
            # Apply individual per-neuron models
            new_post_acts = []
            for d in range(self.num_neurons):
                neuron_input = pre_hist[:, d, :]
                new_post_act = self.neuron_mlps[d](neuron_input)
                new_post_acts.append(new_post_act)
            post_act = torch.cat(new_post_acts, dim=-1)
            
        # Simplified prediction heuristic
        # Random uniform sample of relationships
        idx1 = torch.randint(0, self.num_neurons, (32,), device=x_t.device)
        idx2 = torch.randint(0, self.num_neurons, (32,), device=x_t.device)
        sync_out = post_act[:, idx1] * post_act[:, idx2]
        
        logits = self.output_mlp(sync_out).squeeze(-1)
        
        return logits, (pre_hist, post_act)

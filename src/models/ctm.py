import torch
import torch.nn as nn
from .nlm import NLMGroup

class ContinuousThoughtMachine(nn.Module):
    """
    Continuous Thought Machine (CTM) for streaming reasoning.
    Uses Neural Synchronization as its primary representation.
    """
    def __init__(self, num_neurons, history_len, feature_dim, out_dim):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_len = history_len
        
        # Synapse model: connects current raw features and previous post-activations
        self.synapse = nn.Sequential(
            nn.Linear(num_neurons + feature_dim, num_neurons),
            nn.LayerNorm(num_neurons),
            nn.Tanh()
        )
        
        # Neuron Level Models (NLMs)
        self.nlms = NLMGroup(num_neurons, history_len)
        
        # Output projection from synchronization matrix
        # Since full D x D is large, we project from a flattened upper triangle or sampled pairs
        # Here we use a sampled-pair approach for parameter efficiency
        self.num_sampled_pairs = 128
        self.pair_indices = torch.randint(0, num_neurons, (self.num_sampled_pairs, 2))
        
        self.output_proj = nn.Linear(self.num_sampled_pairs, out_dim)

    def compute_synchronization(self, history_z):
        """
        Computes neural synchronization between sampled pairs.
        history_z shape: (batch_size, num_neurons, total_ticks)
        """
        # (batch_size, num_neurons, total_ticks)
        # Normalize history for stable dot products
        history_z = nn.functional.normalize(history_z, dim=-1)
        
        # Sampled dot products
        u_idx = self.pair_indices[:, 0]
        v_idx = self.pair_indices[:, 1]
        
        # (batch_size, num_sampled_pairs, total_ticks)
        z_u = history_z[:, u_idx, :]
        z_v = history_z[:, v_idx, :]
        
        # Dot product across time dimension
        sync_rep = (z_u * z_v).sum(dim=-1) # (batch_size, num_sampled_pairs)
        return sync_rep

    def forward(self, x, z_prev, hist_pre_act, hist_post_act, ticks=5):
        """
        One external step might involve multiple internal 'thought' ticks.
        x: current features (batch_size, feature_dim)
        z_prev: post-activations from last step (batch_size, num_neurons)
        hist_pre_act: history of pre-activations for NLMs (batch_size, num_neurons, history_len)
        hist_post_act: all history of post-activations for sync (batch_size, num_neurons, total_ticks)
        """
        for _ in range(ticks):
            # 1. Update pre-activations via synapse model
            pre_act = self.synapse(torch.cat([x, z_prev], dim=-1))
            
            # Update pre-activation history (FIFO)
            hist_pre_act = torch.cat([hist_pre_act[:, :, 1:], pre_act.unsqueeze(-1)], dim=-1)
            
            # 2. Update post-activations via NLMs
            z_curr = self.nlms(hist_pre_act)
            
            # Update post-activation history
            hist_post_act = torch.cat([hist_post_act, z_curr.unsqueeze(-1)], dim=-1)
            
            z_prev = z_curr
            
        # 3. Compute synchronization representation
        sync_rep = self.compute_synchronization(hist_post_act)
        
        # 4. Predict
        logits = self.output_proj(sync_rep)
        
        return logits, z_prev, hist_pre_act, hist_post_act

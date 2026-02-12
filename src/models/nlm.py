import torch
import torch.nn as nn

class NeuronLevelModel(nn.Module):
    """
    A mid-level abstraction where each neuron is a private MLP
    processing a history of its incoming pre-activations.
    """
    def __init__(self, history_len, hidden_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(history_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, history):
        """
        history: (batch_size, history_len)
        returns: (batch_size, 1)
        """
        return self.mlp(history)

class NLMGroup(nn.Module):
    """
    Groups multiple NLMs to process them in parallel.
    Uses a 1D convolution as an efficient way to apply per-neuron MLPs.
    """
    def __init__(self, num_neurons, history_len, hidden_dim=16):
        super().__init__()
        self.num_neurons = num_neurons
        self.history_len = history_len
        
        # We can implement this as a Grouped Linear or Grouped Conv1D
        # For simplicity and clarity, we use a Conv1D with groups=num_neurons
        # This effectively gives each 'channel' (neuron) its own weights.
        self.layer1 = nn.Conv1d(
            in_channels=num_neurons,
            out_channels=num_neurons * hidden_dim,
            kernel_size=history_len,
            groups=num_neurons
        )
        self.layer2 = nn.Conv1d(
            in_channels=num_neurons * hidden_dim,
            out_channels=num_neurons,
            kernel_size=1,
            groups=num_neurons
        )

    def forward(self, history):
        """
        history: (batch_size, num_neurons, history_len)
        """
        x = torch.relu(self.layer1(history))
        x = self.layer2(x)
        return x.squeeze(-1) # (batch_size, num_neurons)

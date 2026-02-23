import torch.nn as nn

class StreamingModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def reset_state(self, batch_size=1):
        """
        Return initial state (e.g., None or zero hidden).
        """
        return None
    
    def forward_step(self, x_t, state):
        """
        Process a single step or batch of steps in streaming fashion.
        x_t: (batch, features) or (features,) for single step.
        state: any (hidden state, buffers).
        Returns: (logits, new_state)
        """
        raise NotImplementedError

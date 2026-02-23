import numpy as np
import torch
from torch.utils.data import Dataset

class StreamingSimulator:
    def __init__(self, X, y, batch_size=1):
        """
        X: numpy array of shape (T, F)
        y: numpy array of shape (T,)
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.reset()
    
    def reset(self):
        self.pos = 0
    
    def step(self):
        if self.pos >= len(self.X):
            return None, None  # end of stream
        x_t = self.X[self.pos]
        y_t = self.y[self.pos]
        self.pos += 1
        return x_t, y_t
    
    def step_batch(self):
        if self.pos >= len(self.X):
            return None, None
        end = min(self.pos + self.batch_size, len(self.X))
        X_batch = self.X[self.pos:end]
        y_batch = self.y[self.pos:end]
        self.pos = end
        return X_batch, y_batch

    def done(self):
        return self.pos >= len(self.X)

class WindowDataset(Dataset):
    """Dataset for window-based models like Transformer."""
    def __init__(self, X, y, window=40):
        self.X = X
        self.y = y
        self.window = window
        
    def __len__(self):
        return len(self.X) - self.window + 1
        
    def __getitem__(self, idx):
        x_win = self.X[idx : idx + self.window]
        y_val = self.y[idx + self.window - 1]
        return torch.FloatTensor(x_win), torch.tensor(y_val, dtype=torch.float32)

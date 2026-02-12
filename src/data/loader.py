import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BinanceOrderBookDataset(Dataset):
    """
    Dataset for loading and windowing Binance L2 order book data.
    Enforces clock-time resampling and causal label generation.
    """
    def __init__(self, csv_path, window_size=40, sampling_rate='50ms', horizon='1s', epsilon=0.0):
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.horizon = horizon
        self.epsilon = epsilon
        
        # Load and resample
        df = pd.read_csv(csv_path)
        df['timestamp'] = pd.to_datetime(df['ts_exchange'], unit='ms')
        df = df.set_index('timestamp')
        
        # Resample at clock-time
        self.df_resampled = df.resample(sampling_rate).last().ffill()
        
        # Feature Engineering: Normalized prices and sizes
        self.features = self.construct_features(self.df_resampled)
        
        # Labels: Mid-price direction at horizon H
        self.labels = self.construct_labels(self.df_resampled, horizon)
        
        # Filtering indices where we have a full window and a valid label
        self.valid_indices = self.get_valid_indices()

    def construct_features(self, df):
        """
        Microstructure feature engineering.
        """
        mid = (df['ask_px_1'] + df['bid_px_1']) / 2
        spread = df['ask_px_1'] - df['bid_px_1']
        
        # Log returns
        returns = np.log(mid / mid.shift(1)).fillna(0)
        
        # Imbalance at top 5 levels
        bid_sz_sum = df[[f'bid_sz_{i}' for i in range(1, 6)]].sum(axis=1)
        ask_sz_sum = df[[f'ask_sz_{i}' for i in range(1, 6)]].sum(axis=1)
        imbalance = (bid_sz_sum - ask_sz_sum) / (bid_sz_sum + ask_sz_sum + 1e-8)
        
        # Per-level distances and log-sizes
        feature_list = [returns, spread, imbalance]
        for i in range(1, self.window_size + 1): # Standard practice
             # We actually only want K levels per step
             pass
             
        # Re-implementing more robustly
        feats = []
        feats.append(returns.values)
        feats.append(spread.values)
        feats.append(imbalance.values)
        
        for i in range(1, 11): # K=10 levels
            d_bid = (mid - df[f'bid_px_{i}']) / mid
            d_ask = (df[f'ask_px_{i}'] - mid) / mid
            feats.append(d_bid.values)
            feats.append(d_ask.values)
            feats.append(np.log1p(df[f'bid_sz_{i}'].values))
            feats.append(np.log1p(df[f'ask_sz_{i}'].values))
            
        return np.stack(feats, axis=1) # (N, F)

    def construct_labels(self, df, horizon):
        # Shift mid-price to get future mid
        horizon_steps = int(pd.Timedelta(horizon) / pd.Timedelta(self.sampling_rate))
        mid = (df['ask_px_1'] + df['bid_px_1']) / 2
        future_mid = mid.shift(-horizon_steps)
        
        diff = future_mid - mid
        labels = np.zeros(len(df))
        labels[diff > self.epsilon] = 1
        labels[diff < -self.epsilon] = 0
        return labels

    def get_valid_indices(self):
        # Start after first window, end before horizon
        horizon_steps = int(pd.Timedelta(self.horizon) / pd.Timedelta(self.sampling_rate))
        return np.arange(self.window_size, len(self.df_resampled) - horizon_steps)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        x = self.features[real_idx - self.window_size : real_idx]
        y = self.labels[real_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

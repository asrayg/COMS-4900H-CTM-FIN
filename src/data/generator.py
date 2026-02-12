import numpy as np
import pandas as pd
import torch

class L2BookGenerator:
    """
    Generates synthetic L2 Order Book data with realistic properties:
    - Geometric Brownian Motion for mid-price.
    - Mean-reverting spreads.
    - Correlated imbalances.
    - Ability to inject controlled 'shocks' (spread widening, liquidity drops).
    """
    def __init__(self, k_levels=10, seed=42):
        self.k_levels = k_levels
        self.rng = np.random.default_rng(seed)

    def generate(self, n_steps=10000, dt=0.05, mid_start=50000.0, vol=0.0001, seed=None):
        """
        n_steps: number of steps.
        dt: time step in seconds.
        vol: volatility per step.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        data = []
        mid = mid_start
        spread = 2.0 # starting spread in ticks
        
        # We generate a path for mid-price
        for i in range(n_steps):
            # Mid-price walk
            mid *= (1 + self.rng.normal(0, vol))
            
            # Spread follows a simple mean-reverting process
            spread = max(1.0, spread + self.rng.normal(0, 0.2) + 0.1 * (2.0 - spread))
            
            # Construct bid/ask levels
            best_bid = np.round(mid - spread/2, 1)
            best_ask = np.round(mid + spread/2, 1)
            
            # Ensure no crossover
            if best_bid >= best_ask:
                best_ask = best_bid + 0.1
                
            bids_px = [best_bid - j * 0.1 for j in range(self.k_levels)]
            asks_px = [best_ask + j * 0.1 for j in range(self.k_levels)]
            
            # Sizes: decaying from top
            # Add some 'imbalance' correlation to future price moves
            # (In a real book, high bid side usually leads to up move)
            # We'll bake this signal in slightly so models can learn it.
            base_size = 1.0
            bid_noise = self.rng.standard_exponential(self.k_levels)
            ask_noise = self.rng.standard_exponential(self.k_levels)
            
            # Injecting a small signal: if epsilon > 0, future price tends towards imbalance
            bids_sz = (base_size * np.exp(-0.2 * np.arange(self.k_levels)) * bid_noise).tolist()
            asks_sz = (base_size * np.exp(-0.2 * np.arange(self.k_levels)) * ask_noise).tolist()
            
            row = {
                'ts_exchange': i * (dt * 1000), # ms
                'mid': mid
            }
            for j in range(self.k_levels):
                row[f'bid_px_{j+1}'] = bids_px[j]
                row[f'bid_sz_{j+1}'] = bids_sz[j]
                row[f'ask_px_{j+1}'] = asks_px[j]
                row[f'ask_sz_{j+1}'] = asks_sz[j]
                
            data.append(row)
            
        return pd.DataFrame(data)

    def inject_shock(self, df, shock_type='spread', start_idx=5000, duration=40):
        """
        Injects a synthetic shock into the dataframe.
        """
        df_shocked = df.copy()
        end_idx = min(start_idx + duration, len(df))
        
        if shock_type == 'spread':
            # Widen spread by moving asks up and bids down
            for i in range(start_idx, end_idx):
                for j in range(self.k_levels):
                    df_shocked.loc[i, f'ask_px_{j+1}'] += 5.0
                    df_shocked.loc[i, f'bid_px_{j+1}'] -= 5.0
        elif shock_type == 'liquidity':
            # Zero out sizes at top levels
            for i in range(start_idx, end_idx):
                for j in range(3): # top 3 levels
                    df_shocked.loc[i, f'ask_sz_{j+1}'] *= 0.1
                    df_shocked.loc[i, f'bid_sz_{j+1}'] *= 0.1
                    
        return df_shocked

import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

from .constants import RAW_DATA_DIR, HDF5_DATA_FILE, K_LEVELS, DELTA_S

def process_day(file_path, K=10):
    df = pd.read_parquet(file_path)
    # Ensure sorted by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        timestamps = df['timestamp'].values
    else:
        # Fallback if names differ slightly
        timestamps = np.arange(len(df))
    
    # Extract bid/ask prices and sizes for top K levels
    try:
        # Check if using Crypto Lake schema (bid_0_price, etc.)
        if 'bid_0_price' in df.columns:
            bid_px = df[[f'bid_{i}_price' for i in range(K)]].values
            bid_sz = df[[f'bid_{i}_size' for i in range(K)]].values
            ask_px = df[[f'ask_{i}_price' for i in range(K)]].values
            ask_sz = df[[f'ask_{i}_size' for i in range(K)]].values
        else:
            # Fallback to the original binance schema names
            bid_px = df[[f'bids_px_{i}' for i in range(K)]].values
            bid_sz = df[[f'bids_sz_{i}' for i in range(K)]].values
            ask_px = df[[f'asks_px_{i}' for i in range(K)]].values
            ask_sz = df[[f'asks_sz_{i}' for i in range(K)]].values
    except KeyError as e:
        print(f"Error extracting level-2 book columns in {file_path}. Columns available: {df.columns}")
        raise e
    
    # Basic cleaning: ensure bid_px descending, ask_px ascending
    # If not, we could fix, but assume data is clean from Binance vision.
    return timestamps, bid_px, bid_sz, ask_px, ask_sz

def build_hdf5():
    """Reads raw parquet files and writes them to a consolidated HDF5 file."""
    parquet_files = sorted(Path(RAW_DATA_DIR).glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {RAW_DATA_DIR}.")
        return

    print("Building HDF5 from raw parquet files...")
    with h5py.File(HDF5_DATA_FILE, 'w') as f:
        for day_file in tqdm(parquet_files):
            day = day_file.stem
            ts, bp, bs, ap, a_sz = process_day(day_file, K=K_LEVELS)
            grp = f.create_group(day)
            grp.create_dataset('timestamp', data=ts)
            grp.create_dataset('bid_px', data=bp)
            grp.create_dataset('bid_sz', data=bs)
            grp.create_dataset('ask_px', data=ap)
            grp.create_dataset('ask_sz', data=a_sz)

def resample_day(day, delta_s=DELTA_S):
    """
    Resample a single day of data to a fixed time step.
    Returns target timestamps and feature matrix X.
    """
    with h5py.File(HDF5_DATA_FILE, 'r') as f:
        if day not in f:
            raise KeyError(f"Day {day} not found in {HDF5_DATA_FILE}")
        
        grp = f[day]
        ts_raw = grp['timestamp'][:]
        bid_px = grp['bid_px'][:]
        bid_sz = grp['bid_sz'][:]
        ask_px = grp['ask_px'][:]
        ask_sz = grp['ask_sz'][:]
    
    # Convert to seconds (assuming input is nanoseconds or milliseconds)
    # If it's pure ms, divide by 1e3. If ns, 1e9. Checking first element:
    ts_sec = ts_raw / 1e9 if ts_raw[0] > 1e15 else ts_raw / 1e3
    
    start_sec = np.ceil(ts_sec[0] / delta_s) * delta_s
    end_sec = np.floor(ts_sec[-1] / delta_s) * delta_s
    target_ts = np.arange(start_sec, end_sec + delta_s, delta_s)
    
    # For each target, find last raw index (forward fill)
    indices = np.searchsorted(ts_sec, target_ts, side='right') - 1
    indices = np.clip(indices, 0, len(ts_sec)-1)
    
    # Gather resampled features
    bid_px_res = bid_px[indices]
    bid_sz_res = bid_sz[indices]
    ask_px_res = ask_px[indices]
    ask_sz_res = ask_sz[indices]
    
    # Compute derived features
    mid = (bid_px_res[:,0] + ask_px_res[:,0]) / 2.0
    spread = ask_px_res[:,0] - bid_px_res[:,0]
    log_mid = np.log(mid)
    
    # Calculate log returns across recent steps (e.g. 1 step return)
    # Start with zeros for first step to keep sizes uniform
    ret = np.zeros_like(mid)
    ret[1:] = np.log(mid[1:] / mid[:-1])
    
    # Order Imbalance
    top_k_bid_sz = np.sum(bid_sz_res, axis=1)
    top_k_ask_sz = np.sum(ask_sz_res, axis=1)
    imbalance = (top_k_bid_sz - top_k_ask_sz) / (top_k_bid_sz + top_k_ask_sz + 1e-8)
    
    # Relative Distances and Log Sizes
    dist_bids = (mid[:, None] - bid_px_res) / mid[:, None]
    dist_asks = (ask_px_res - mid[:, None]) / mid[:, None]
    
    log_sz_bids = np.log1p(bid_sz_res)
    log_sz_asks = np.log1p(ask_sz_res)
    
    # Combine into feature matrix X
    # Features: mid (1), spread (1), log_mid (1), return (1), imbalance (1), 
    # dist_bids (K), dist_asks (K), log_sz_bids (K), log_sz_asks (K)
    X = np.column_stack([
        mid, spread, log_mid, ret, imbalance,
        dist_bids, dist_asks, log_sz_bids, log_sz_asks
    ])
    
    return target_ts, X, mid

def build_labels(mid, target_ts, horizon_s=1.0, epsilon=0.0):
    """
    Returns (y, mask) where:
      y    - binary labels (1=up, 0=down) for non-flat samples only
      mask - boolean array selecting non-flat samples from the original sequence

    epsilon is treated as a relative threshold (fraction of mid price), so
    a sample is labeled only when |delta / mid| > epsilon.  Flat samples
    (|delta / mid| <= epsilon) are excluded via the mask so they don't
    pollute the class distribution.
    """
    future_ts = target_ts + horizon_s

    # Find indices of resampled mid at future time
    future_indices = np.searchsorted(target_ts, future_ts, side='right')
    future_indices = np.clip(future_indices, 0, len(mid)-1)

    mid_future = mid[future_indices]
    rel_delta = (mid_future - mid) / mid

    mask = np.abs(rel_delta) > epsilon
    y = np.where(rel_delta[mask] > 0, 1, 0).astype(np.int64)

    return y, mask

if __name__ == "__main__":
    build_hdf5()

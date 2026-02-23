import pandas as pd
from sortedcontainers import SortedDict
import time
import argparse
from pathlib import Path
import csv

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
K_LEVELS = 10
INTERVAL_US = 50 * 1000 # 50 milliseconds

def build_snapshots(csv_path):
    print(f"Rebuilding orderbook from {csv_path}...")
    print(f"Target Snapshot Frequency: {int(INTERVAL_US/1000)}ms")
    
    bids = SortedDict()
    asks = SortedDict()
    snapshots = []
    
    current_interval_ts = None
    row_count = 0
    start_time = time.time()
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Fast indexing
        idx_ts = header.index('timestamp')
        idx_side = header.index('side')
        idx_price = header.index('price')
        idx_amount = header.index('amount')
        
        for row in reader:
            ts = int(row[idx_ts])
            
            if current_interval_ts is None:
                current_interval_ts = ts - (ts % INTERVAL_US) + INTERVAL_US
                
            # Snapshot check (if time advanced past our interval)
            while ts >= current_interval_ts:
                if len(bids) > 0 and len(asks) > 0:
                    snap = [current_interval_ts / 1000.0] # timestamp in ms
                    
                    # Top K Bids (prices are negative in SortedDict)
                    bid_iter = bids.items()
                    for i in range(K_LEVELS):
                        if i < len(bid_iter):
                            snap.append(-bid_iter[i][0])
                            snap.append(bid_iter[i][1])
                        else:
                            snap.append(0.0)
                            snap.append(0.0)
                            
                    # Top K Asks
                    ask_iter = asks.items()
                    for i in range(K_LEVELS):
                        if i < len(ask_iter):
                            snap.append(ask_iter[i][0])
                            snap.append(ask_iter[i][1])
                        else:
                            snap.append(0.0)
                            snap.append(0.0)
                            
                    snapshots.append(snap)
                current_interval_ts += INTERVAL_US
                
            # Apply update
            price = float(row[idx_price])
            amount = float(row[idx_amount])
            
            if row[idx_side] == 'bid':
                if amount == 0:
                    bids.pop(-price, None)
                else:
                    bids[-price] = amount
            elif row[idx_side] == 'ask':
                if amount == 0:
                    asks.pop(price, None)
                else:
                    asks[price] = amount
                    
            row_count += 1
            if row_count % 5000000 == 0:
                print(f"Processed {row_count:,} rows in {time.time() - start_time:.1f}s...")
                
    print(f"Parsing complete. Captured {len(snapshots):,} snapshots.")
    
    # Define columns
    cols = ['timestamp']
    for i in range(K_LEVELS):
        cols.extend([f'bid_{i}_price', f'bid_{i}_size'])
    for i in range(K_LEVELS):
        cols.extend([f'ask_{i}_price', f'ask_{i}_size'])
        
    df = pd.DataFrame(snapshots, columns=cols)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Tardis L2 Orderbook Rebuilder")
    parser.add_argument('csv_file', type=str, help="Path to raw Tardis incremental CSV")
    args = parser.parse_args()
    
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"File {csv_path} not found.")
        exit(1)
        
    df = build_snapshots(csv_path)
    
    out_name = RAW_DATA_DIR / f"tardis_rebuilt_{csv_path.stem}.parquet"
    df.to_parquet(out_name)
    print(f"Successfully rebuilt and saved to {out_name}")

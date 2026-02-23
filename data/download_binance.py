import os
import time
import argparse
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def fetch_historical_binance_futures(symbol, start_date, end_date):
    """
    Downloads historical daily orderbook depth from Binance Public Data API
    Specifically targets the Futures (UM) market as Spot does not store L2 depth.
    (zip containing csv) and converts to parquet.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = start_date
    
    print(f"Fetching historical Futures L2 data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        
        # NOTE: Binance Spot does not have bookDepth. We MUST use Futures UM (USDS-Margined).
        url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date_str}.zip"
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            print(f"[{date_str}] Download successful. Extracting...")
            try:
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    csv_filename = z.namelist()[0]
                    with z.open(csv_filename) as f:
                        # Binance Futures bookDepth CSV format 
                        df = pd.read_csv(f)
                        out_file = RAW_DATA_DIR / f"{symbol}_{date_str}.parquet"
                        df.to_parquet(out_file)
                        print(f"[{date_str}] Successfully saved to {out_file.name}")
            except Exception as e:
                print(f"[{date_str}] Error extracting/converting: {e}")
        else:
            print(f"[{date_str}] Download failed (HTTP {response.status_code}). URL tried: {url}")
            
        current += timedelta(days=1)
        time.sleep(1) # Rate limit protection

def collect_live_websocket(symbol, duration_minutes):
    """
    Connects to the Binance WebSocket API using python-binance and records 
    high-resolution orderbook depth snapshots continuously.
    """
    from binance.client import Client
    from binance.depthcache import ThreadedDepthCacheManager
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    binance_symbol = symbol.replace('-', '') # e.g. BTC-USDT to BTCUSDT
    print(f"Starting live WebSocket data collection for {binance_symbol}...")
    print(f"Collecting snapshots every 100ms for {duration_minutes} minutes.")
    
    # Initialize python-binance depth cache
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    try:
        client = Client(api_key, api_secret)
        dcm = ThreadedDepthCacheManager(api_key, api_secret)
    except Exception as e:
        print("Error initializing Binance DepthCacheManager:", e)
        return
        
    # Start the manager
    dcm.start()
    
    def process_depth(depth_cache):
        pass
        
    dcm.start_depth_cache(process_depth, symbol=binance_symbol)
    time.sleep(3) # Wait for initial cache sync
    
    snapshots = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            depth = dcm.get_depth_cache(binance_symbol)
            if depth:
                bids = depth.get_bids()[:10]
                asks = depth.get_asks()[:10]
                
                row = {'received_time': int(time.time() * 1000)}
                for i in range(10):
                    if i < len(bids):
                        row[f'bid_{i}_price'] = float(bids[i][0])
                        row[f'bid_{i}_size'] = float(bids[i][1])
                    else:
                        row[f'bid_{i}_price'] = 0.0
                        row[f'bid_{i}_size'] = 0.0
                        
                    if i < len(asks):
                        row[f'ask_{i}_price'] = float(asks[i][0])
                        row[f'ask_{i}_size'] = float(asks[i][1])
                    else:
                        row[f'ask_{i}_price'] = 0.0
                        row[f'ask_{i}_size'] = 0.0
                        
                snapshots.append(row)
                
            time.sleep(0.1) # 100ms sampling latency
            
            if len(snapshots) % 600 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Live collection status: {len(snapshots)} snapshots... ({elapsed:.1f} / {duration_minutes} min)")
                
    except KeyboardInterrupt:
        print("Live collection interrupted by user.")
    
    finally:
        dcm.stop()
        if snapshots:
            df = pd.DataFrame(snapshots)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file = RAW_DATA_DIR / f"{symbol}_live_{timestamp_str}.parquet"
            df.to_parquet(out_file)
            print(f"Saved {len(df)} live snapshots to {out_file.name}")
        else:
            print("No snapshots were collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orderbook Depth Data Downloader")
    parser.add_argument('--mode', type=str, choices=['historical', 'live'], required=True, 
                        help="historical (Binance Futures S3 buckets) or live (Binance WebSocket).")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help="Symbol format: BTCUSDT")
    parser.add_argument('--start', type=str, help='Start Date for historical: YYYY-MM-DD')
    parser.add_argument('--end', type=str, help='End Date for historical: YYYY-MM-DD')
    parser.add_argument('--duration', type=int, default=60, help='Duration in minutes for live WebSocket scraping.')
    
    args = parser.parse_args()
    
    if args.mode == 'historical':
        if not args.start or not args.end:
            print("Error: --start and --end are required for historical mode.")
            exit(1)
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        fetch_historical_binance_futures(args.symbol, start_dt, end_dt)
    elif args.mode == 'live':
        collect_live_websocket(args.symbol, args.duration)

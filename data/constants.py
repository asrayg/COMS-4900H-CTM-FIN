import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
HDF5_DATA_FILE = DATA_DIR / 'btc_book.h5'

# Make sure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Order Book Constants
K_LEVELS = 10        # Depth of order book to consider

# Resampling Constants
DELTA_MS = 50        # Target resampling interval in milliseconds (default: 50)
DELTA_S = DELTA_MS / 1000.0
HORIZON_S = 1.0      # Prediction horizon in seconds

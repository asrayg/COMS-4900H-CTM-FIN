import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from shock_tests.inject_shocks import apply_spread_shock, apply_liquidity_cliff
from data.dataset import StreamingSimulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shock_type', type=str, choices=['spread', 'liquidity'], default='spread')
    args = parser.parse_args()
    
    print(f"Running robustness evaluation with shock type: {args.shock_type}")
    print("Models will be evaluated on cleanly recovered sequence vs shocked sequence.")
    
    # Implementation Steps:
    # 1. Load clean data segment
    # 2. Inject shock into raw arrays using inject_shocks.py
    # 3. Simulate through the models
    # 4. Measure recovery time, flip rates, and probability deviation during shock
    
if __name__ == "__main__":
    main()

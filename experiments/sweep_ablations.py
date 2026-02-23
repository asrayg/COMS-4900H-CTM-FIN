import subprocess
import yaml
from pathlib import Path
import os

BASE_CONFIG = 'configs/default.yaml'

def run_sweep():
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    variations = [
        {'delta_ms': 50},
        {'delta_ms': 10},
        {'horizon_s': 1.0},
        {'horizon_s': 0.5},
        {'epsilon': 0.0},
        {'epsilon': 0.5},
    ]

    for i, var in enumerate(variations):
        config = base.copy()
        config.update(var)
        temp_config = f'configs/temp_ablation_{i}.yaml'
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        print(f"\\n--- Running ablation {i} with config modifications: {var} ---")
        subprocess.run(['python', 'experiments/run_experiment.py', '--config', temp_config])
        
        # Cleanup
        os.remove(temp_config)
        
if __name__ == "__main__":
    run_sweep()

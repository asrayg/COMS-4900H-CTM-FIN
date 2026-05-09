"""
CTM ablation sweep: varies internal ticks, sync pairs, resampling frequency,
and prediction horizon on CTM-Inspired.

Results saved to experiments/results/ablation_results.json.

Usage:
    python experiments/sweep_ablations.py              # full sweep
    python experiments/sweep_ablations.py --fast       # quick debug
"""
import argparse
import json
import yaml
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_experiment import load_data, run_single_experiment

# Ablation grid: each entry is (param_name, config_key, values)
ABLATION_GRID = [
    ('internal_ticks', 'max_ticks', [1, 2, 5, 10]),
    ('sync_pairs',     'num_pairs', [32, 64, 128, 256]),
    ('delta_ms',       'delta_ms',  [10, 50]),
    ('horizon_s',      'horizon_s', [0.5, 1.0, 5.0]),
]


def main():
    parser = argparse.ArgumentParser(description="CTM ablation sweep")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--fast', action='store_true',
                        help='Debug mode: 10k samples, 2 epochs')
    parser.add_argument('--fast-samples', type=int, default=10_000)
    args = parser.parse_args()

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    if args.fast:
        base_config['epochs'] = min(base_config['epochs'], 2)
        print(f"[fast mode] capping to {args.fast_samples} samples, {base_config['epochs']} epochs")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'ablation_results.json'

    ablation_results = []

    for param_name, config_key, values in ABLATION_GRID:
        # For delta_ms and horizon_s, data changes per value so we must reload
        data_varies = config_key in ('delta_ms', 'horizon_s')

        # Pre-load data for non-data-varying params using base config
        if not data_varies:
            X_all, y_all = load_data(base_config, fast=args.fast,
                                     fast_samples=args.fast_samples)

        for val in values:
            print(f"\n{'='*60}")
            print(f"  Ablation: {param_name} = {val}")
            print(f"{'='*60}")

            config = base_config.copy()
            config['model_type'] = 'ctm_inspired'
            config[config_key] = val

            try:
                if data_varies:
                    X_all, y_all = load_data(config, fast=args.fast,
                                             fast_samples=args.fast_samples)

                model, metrics, _ = run_single_experiment(config, X_all, y_all, device)

                entry = {
                    'param': param_name,
                    'value': val,
                    'accuracy': float(metrics['accuracy']),
                    'auroc': float(metrics['auroc']),
                    'flip_rate': float(metrics['flip_rate']),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                entry = {
                    'param': param_name,
                    'value': val,
                    'error': str(e),
                }

            ablation_results.append(entry)

            # Save incrementally
            with open(results_file, 'w') as f:
                json.dump(ablation_results, f, indent=2)

            print(f"  Result: {entry}")

    print(f"\nAblation sweep complete. Results saved to {results_file}")


if __name__ == "__main__":
    main()

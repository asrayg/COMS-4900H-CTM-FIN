"""
Unified experiment runner: trains + evaluates all 5 model types sequentially.
Results are saved to experiments/results/main_results.json.

Usage:
    python experiments/run_all.py              # full run
    python experiments/run_all.py --fast       # quick debug (10k samples, 2 epochs)
"""
import argparse
import json
import yaml
import sys
import os
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_experiment import load_data, run_single_experiment

ALL_MODEL_TYPES = ['logistic', 'lstm', 'transformer', 'ctm_inspired', 'ctm_full']

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate all models")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--fast', action='store_true',
                        help='Debug mode: 10k samples, 2 epochs')
    parser.add_argument('--fast-samples', type=int, default=10_000)
    parser.add_argument('--models', nargs='+', default=ALL_MODEL_TYPES,
                        choices=ALL_MODEL_TYPES,
                        help='Subset of models to run (default: all)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.fast:
        config['epochs'] = min(config['epochs'], 2)
        print(f"[fast mode] capping to {args.fast_samples} samples, {config['epochs']} epochs")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data once for all models
    X_all, y_all = load_data(config, fast=args.fast, fast_samples=args.fast_samples)
    print(f"Total samples: {len(X_all)}, features: {X_all.shape[1]}")

    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'main_results.json'

    all_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_results = json.load(f)

    for mtype in args.models:
        print(f"\n{'='*60}")
        print(f"  Training model: {mtype}")
        print(f"{'='*60}")

        model_config = config.copy()
        model_config['model_type'] = mtype

        try:
            model, metrics, ckpt_path = run_single_experiment(
                model_config, X_all, y_all, device)

            all_results[mtype] = {
                'accuracy': float(metrics['accuracy']),
                'auroc': float(metrics['auroc']),
                'flip_rate': float(metrics['flip_rate']),
            }

            # Save incrementally after each model
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

            print(f"  {mtype} done: acc={metrics['accuracy']:.4f} "
                  f"auroc={metrics['auroc']:.4f} flip={metrics['flip_rate']:.4f}")
        except Exception as e:
            print(f"  ERROR training {mtype}: {e}")
            all_results[mtype] = {'error': str(e)}
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {results_file}")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()

import argparse
import json
import yaml
import sys
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.logistic import LogisticStream
from models.lstm import LSTMStream
from models.transformer import TransformerWindow
from models.ctm_inspired import CTMInspired
from models.ctm_full import CTMFull

from training.trainer import StreamingTrainer
from training.evaluate import evaluate_model
from data.dataset import StreamingSimulator
from data.preprocess import resample_day, build_labels

CTM_MODEL_TYPES = {'ctm_inspired', 'ctm_full'}

def get_model(config):
    mtype = config['model_type']
    if mtype == 'logistic':
        return LogisticStream(config['input_dim'], config['hidden_dim'])
    elif mtype == 'lstm':
        return LSTMStream(config['input_dim'], config['hidden_dim'])
    elif mtype == 'transformer':
        return TransformerWindow(config['input_dim'], config['hidden_dim'])
    elif mtype == 'ctm_inspired':
        return CTMInspired(config['input_dim'], config['hidden_dim'],
                           config.get('history_len', 64),
                           config.get('num_pairs', 128),
                           config.get('max_ticks', 5))
    elif mtype == 'ctm_full':
        return CTMFull(config['input_dim'], config['hidden_dim'])
    else:
        raise ValueError(f"Unknown model type {mtype}")


def load_data(config, fast=False, fast_samples=10_000):
    """Load and preprocess all data from HDF5. Returns X_all, y_all."""
    delta_s = config['delta_ms'] / 1000.0
    horizon_s = config['horizon_s']
    epsilon = config['epsilon']

    X_list = []
    y_list = []

    with h5py.File(config['data_path'], 'r') as f:
        days = sorted(f.keys())

    print(f"Found {len(days)} day(s) in {config['data_path']}")

    for day in tqdm(days, desc="Loading days"):
        target_ts, X, mid = resample_day(day, delta_s)
        y, mask = build_labels(mid, target_ts, horizon_s, epsilon)
        X_list.append(X[mask])
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if fast:
        X_all = X_all[:fast_samples]
        y_all = y_all[:fast_samples]

    # Z-score normalization (critical for numerical stability)
    mean = X_all.mean(axis=0)
    std = X_all.std(axis=0) + 1e-8
    X_all = (X_all - mean) / std

    return X_all, y_all


def run_single_experiment(config, X_all, y_all, device='cpu'):
    """Train and evaluate a single model. Returns (model, metrics, best_model_path)."""
    model = get_model(config).to(device)
    mtype = config['model_type']
    use_dual_loss = mtype in CTM_MODEL_TYPES

    print(f"Initialized {mtype} on {device} (dual_loss={use_dual_loss})")

    # Train/val split (chronological 80/20)
    split_idx = int(0.8 * len(X_all))
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_val, y_val = X_all[split_idx:], y_all[split_idx:]

    train_sim = StreamingSimulator(X_train, y_train)
    val_sim = StreamingSimulator(X_val, y_val)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    trainer = StreamingTrainer(model, train_sim, val_sim, optimizer, criterion, device,
                               use_dual_loss=use_dual_loss)

    os.makedirs('models', exist_ok=True)
    best_val_acc = 0
    patience_counter = 0
    best_model_path = f"models/{mtype}_best.pt"

    epoch_bar = tqdm(range(config['epochs']), desc=f"Epochs ({mtype})")
    for epoch in epoch_bar:
        train_loss = trainer.train_epoch()
        val_acc = trainer.validate()
        epoch_bar.set_postfix(loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}",
                              best=f"{best_val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                tqdm.write(f"Early stopping for {mtype}")
                break

    # Final evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    metrics = evaluate_model(model, val_sim, device)
    print(f"{mtype} results: {metrics}")

    return model, metrics, best_model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--fast', action='store_true',
                        help='Debug mode: cap to 10k samples and 2 epochs for quick iteration')
    parser.add_argument('--fast-samples', type=int, default=10_000,
                        help='Number of samples to use in --fast mode (default: 10000)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.fast:
        config['epochs'] = min(config['epochs'], 2)
        print(f"[fast mode] capping to {args.fast_samples} samples, {config['epochs']} epochs")

    print(f"Starting experiment with config: {config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X_all, y_all = load_data(config, fast=args.fast, fast_samples=args.fast_samples)
    print(f"Total samples: {len(X_all)}, features: {X_all.shape[1]}")

    model, metrics, _ = run_single_experiment(config, X_all, y_all, device)

    # Save results JSON
    results_dir = Path(__file__).resolve().parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'main_results.json'

    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    existing[config['model_type']] = {
        'accuracy': float(metrics['accuracy']),
        'auroc': float(metrics['auroc']),
        'flip_rate': float(metrics['flip_rate']),
    }

    with open(results_file, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()

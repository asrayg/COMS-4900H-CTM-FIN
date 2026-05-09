"""
Shock evaluation: loads pre-trained model checkpoints, runs clean and shocked
evaluation segments, and computes shock-specific metrics.

Results saved to experiments/results/shock_results.json.

Usage:
    python shock_tests/run_shock_eval.py --shock_type spread
    python shock_tests/run_shock_eval.py --shock_type liquidity
    python shock_tests/run_shock_eval.py --shock_type all
"""
import argparse
import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import yaml

from shock_tests.inject_shocks import apply_spread_shock, apply_liquidity_cliff
from data.dataset import StreamingSimulator
from data.preprocess import build_labels
from data.constants import HDF5_DATA_FILE, K_LEVELS
from experiments.run_experiment import get_model

ALL_MODEL_TYPES = ['logistic', 'lstm', 'transformer', 'ctm_inspired', 'ctm_full']
SHOCK_DURATION = 40
RECOVERY_WINDOW = 20
RECOVERY_THRESHOLD = 0.01  # within 1% of pre-shock accuracy


def compute_features_from_book(bid_px, bid_sz, ask_px, ask_sz):
    """Recompute the same features as preprocess.resample_day from raw book arrays."""
    mid = (bid_px[:, 0] + ask_px[:, 0]) / 2.0
    spread = ask_px[:, 0] - bid_px[:, 0]
    log_mid = np.log(np.clip(mid, 1e-8, None))

    ret = np.zeros_like(mid)
    ret[1:] = np.log(np.clip(mid[1:] / np.clip(mid[:-1], 1e-8, None), 1e-8, None))

    top_k_bid_sz = np.sum(bid_sz, axis=1)
    top_k_ask_sz = np.sum(ask_sz, axis=1)
    imbalance = (top_k_bid_sz - top_k_ask_sz) / (top_k_bid_sz + top_k_ask_sz + 1e-8)

    dist_bids = (mid[:, None] - bid_px) / np.clip(mid[:, None], 1e-8, None)
    dist_asks = (ask_px - mid[:, None]) / np.clip(mid[:, None], 1e-8, None)

    log_sz_bids = np.log1p(bid_sz)
    log_sz_asks = np.log1p(ask_sz)

    X = np.column_stack([
        mid, spread, log_mid, ret, imbalance,
        dist_bids, dist_asks, log_sz_bids, log_sz_asks
    ])
    return X, mid


def load_val_segment(config, n_samples=2000):
    """Load the last n_samples from the dataset as raw book arrays + labels."""
    delta_s = config['delta_ms'] / 1000.0

    with h5py.File(config['data_path'], 'r') as f:
        days = sorted(f.keys())
        # Use last day
        day = days[-1]
        grp = f[day]
        ts_raw = grp['timestamp'][:]
        bid_px = grp['bid_px'][:]
        bid_sz = grp['bid_sz'][:]
        ask_px = grp['ask_px'][:]
        ask_sz = grp['ask_sz'][:]

    # Resample to fixed intervals
    ts_sec = ts_raw / 1e9 if ts_raw[0] > 1e15 else ts_raw / 1e3
    start_sec = np.ceil(ts_sec[0] / delta_s) * delta_s
    end_sec = np.floor(ts_sec[-1] / delta_s) * delta_s
    target_ts = np.arange(start_sec, end_sec + delta_s, delta_s)

    indices = np.searchsorted(ts_sec, target_ts, side='right') - 1
    indices = np.clip(indices, 0, len(ts_sec) - 1)

    bid_px_res = bid_px[indices]
    bid_sz_res = bid_sz[indices]
    ask_px_res = ask_px[indices]
    ask_sz_res = ask_sz[indices]

    # Take last n_samples
    bid_px_res = bid_px_res[-n_samples:]
    bid_sz_res = bid_sz_res[-n_samples:]
    ask_px_res = ask_px_res[-n_samples:]
    ask_sz_res = ask_sz_res[-n_samples:]
    target_ts = target_ts[-n_samples:]

    return target_ts, bid_px_res, bid_sz_res, ask_px_res, ask_sz_res


def streaming_predict(model, X, device='cpu', desc="Predicting"):
    """Run model in streaming mode, return per-step logits."""
    model.eval()
    state = model.reset_state(batch_size=1)
    all_logits = []

    with torch.no_grad():
        for t in tqdm(range(len(X)), desc=desc, leave=False, unit="step"):
            x_t = torch.FloatTensor(X[t]).unsqueeze(0).to(device)
            logits, state = model.forward_step(x_t, state)
            if isinstance(logits, list):
                logits = logits[-1]
            all_logits.append(logits.cpu().item())

    return np.array(all_logits)


def compute_shock_metrics(clean_logits, shocked_logits, y, shock_start, shock_end):
    """Compute shock-specific metrics."""
    clean_preds = (clean_logits > 0).astype(int)
    shocked_preds = (shocked_logits > 0).astype(int)

    # Pre-shock baseline accuracy using clean predictions before shock
    if shock_start > 0:
        pre_shock_acc = np.mean(clean_preds[:shock_start] == y[:shock_start])
    else:
        pre_shock_acc = 0.5

    # Accuracy during shock window
    shock_slice = slice(shock_start, shock_end)
    shock_acc = np.mean(shocked_preds[shock_slice] == y[shock_slice])

    # Flip rate during shock window
    shock_preds_window = shocked_preds[shock_start:shock_end]
    if len(shock_preds_window) > 1:
        shock_flip_rate = float(np.sum(shock_preds_window[1:] != shock_preds_window[:-1])) / len(shock_preds_window)
    else:
        shock_flip_rate = 0.0

    # Recovery time: steps after shock ends until rolling accuracy of
    # the SHOCKED model returns within threshold of pre-shock clean accuracy.
    n_total = len(shocked_preds)
    post_shock_start = shock_end

    if post_shock_start + RECOVERY_WINDOW < n_total:
        recovered = False
        for t in range(post_shock_start, n_total - RECOVERY_WINDOW):
            window_preds = shocked_preds[t:t + RECOVERY_WINDOW]
            window_y = y[t:t + RECOVERY_WINDOW]
            rolling_acc = np.mean(window_preds == window_y)
            if abs(rolling_acc - pre_shock_acc) <= RECOVERY_THRESHOLD:
                recovery_steps = t - post_shock_start
                recovered = True
                break
        if not recovered:
            recovery_steps = n_total - post_shock_start  # never recovered
    else:
        recovery_steps = -1  # not enough post-shock data

    # Also report clean accuracy during the same window for comparison
    clean_shock_acc = np.mean(clean_preds[shock_slice] == y[shock_slice])

    return {
        'pre_shock_accuracy': float(pre_shock_acc),
        'clean_shock_accuracy': float(clean_shock_acc),
        'shock_accuracy': float(shock_acc),
        'shock_flip_rate': float(shock_flip_rate),
        'recovery_steps': int(recovery_steps),
    }


def run_shock_eval(config, shock_type, device='cpu'):
    """Run shock evaluation for all models and a given shock type."""
    print(f"\nLoading validation segment...")
    target_ts, bid_px, bid_sz, ask_px, ask_sz = load_val_segment(config)
    n_raw = len(bid_px)
    print(f"  Raw segment: {n_raw} steps")

    # Compute clean features and labels
    X_clean, mid_clean = compute_features_from_book(bid_px, bid_sz, ask_px, ask_sz)
    y_full, mask = build_labels(mid_clean, target_ts, config['horizon_s'], config['epsilon'])
    print(f"  After epsilon filter: {mask.sum()} / {len(mask)} steps retained")

    # Apply shock to raw book arrays BEFORE feature computation.
    # The shock is applied at the center of the raw segment.
    if shock_type == 'spread':
        s_bid, s_ask, s_bsz, s_asz = apply_spread_shock(
            bid_px, ask_px, bid_sz, ask_sz, duration_steps=SHOCK_DURATION)
    elif shock_type == 'liquidity':
        s_bid, s_ask, s_bsz, s_asz = apply_liquidity_cliff(
            bid_px, ask_px, bid_sz, ask_sz, duration_steps=SHOCK_DURATION)
    else:
        raise ValueError(f"Unknown shock type: {shock_type}")

    # Recompute features on shocked data (note: spread shock returns bid, ask, bid_sz, ask_sz)
    X_shocked, _ = compute_features_from_book(s_bid, s_bsz, s_ask, s_asz)

    # Apply the same mask to both clean and shocked features
    X_clean_masked = X_clean[mask]
    X_shocked_masked = X_shocked[mask]

    # Map the raw shock window into masked-index space.
    # The raw shock is at center of the raw segment:
    raw_center = n_raw // 2
    raw_shock_start = max(0, raw_center - SHOCK_DURATION // 2)
    raw_shock_end = min(n_raw, raw_center + SHOCK_DURATION // 2)

    # Convert raw indices to masked indices
    mask_cumsum = np.cumsum(mask)
    shock_start = int(mask_cumsum[raw_shock_start]) if raw_shock_start < len(mask) else 0
    shock_end = int(mask_cumsum[min(raw_shock_end, len(mask) - 1)])
    # Ensure non-empty window
    if shock_end <= shock_start:
        shock_end = min(shock_start + SHOCK_DURATION, len(X_clean_masked))

    print(f"  Shock window (masked): [{shock_start}, {shock_end}) "
          f"({shock_end - shock_start} steps)")

    results = {}

    for mtype in ALL_MODEL_TYPES:
        ckpt_path = f"models/{mtype}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {mtype}: no checkpoint at {ckpt_path}")
            results[mtype] = {'error': 'no checkpoint'}
            continue

        print(f"  Evaluating {mtype} under {shock_type} shock...")

        model_config = config.copy()
        model_config['model_type'] = mtype
        model = get_model(model_config)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)

        clean_logits = streaming_predict(model, X_clean_masked, device,
                                          desc=f"    {mtype} clean")
        shocked_logits = streaming_predict(model, X_shocked_masked, device,
                                            desc=f"    {mtype} shocked")

        metrics = compute_shock_metrics(clean_logits, shocked_logits, y_full,
                                         shock_start, shock_end)
        results[mtype] = metrics

        # Diagnostic: check if model is predicting constant class
        clean_preds = (clean_logits > 0).astype(int)
        unique_preds = np.unique(clean_preds)
        if len(unique_preds) == 1:
            print(f"    WARNING: {mtype} predicts constant class {unique_preds[0]} "
                  f"(likely undertrained)")

        print(f"    {mtype}: pre_acc={metrics['pre_shock_accuracy']:.4f} "
              f"clean_shock_acc={metrics['clean_shock_accuracy']:.4f} "
              f"shock_acc={metrics['shock_accuracy']:.4f} "
              f"flip={metrics['shock_flip_rate']:.4f} "
              f"recovery={metrics['recovery_steps']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Shock robustness evaluation")
    parser.add_argument('--shock_type', type=str,
                        choices=['spread', 'liquidity', 'all'], default='spread')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    results_dir = Path(__file__).resolve().parent.parent / 'experiments' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / 'shock_results.json'

    all_shock_results = {}
    if results_file.exists():
        with open(results_file) as f:
            all_shock_results = json.load(f)

    shock_types = ['spread', 'liquidity'] if args.shock_type == 'all' else [args.shock_type]

    for st in shock_types:
        print(f"\n{'='*60}")
        print(f"  Shock type: {st}")
        print(f"{'='*60}")

        results = run_shock_eval(config, st, device)
        all_shock_results[st] = results

        with open(results_file, 'w') as f:
            json.dump(all_shock_results, f, indent=2)

    print(f"\nShock results saved to {results_file}")


if __name__ == "__main__":
    main()

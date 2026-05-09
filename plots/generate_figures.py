"""
Generate publication figures from experiment result JSON files.

Usage:
    python plots/generate_figures.py
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / 'experiments' / 'results'
PLOTS_DIR = BASE_DIR / 'plots'


def load_json(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        print(f"  Warning: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def plot_anytime_curve():
    """Plots AUROC vs number of internal CTM ticks from ablation results."""
    data = load_json('ablation_results.json')
    if data is None:
        return

    # Filter for internal_ticks ablation
    tick_entries = [e for e in data if e.get('param') == 'internal_ticks' and 'auroc' in e]
    if not tick_entries:
        print("  No internal_ticks ablation data found.")
        return

    ticks = [e['value'] for e in tick_entries]
    aurocs = [e['auroc'] for e in tick_entries]

    # Also load transformer baseline from main results
    main = load_json('main_results.json')
    transformer_auroc = None
    if main and 'transformer' in main and 'auroc' in main['transformer']:
        transformer_auroc = main['transformer']['auroc']

    plt.figure(figsize=(8, 5))
    plt.plot(ticks, aurocs, marker='o', linewidth=2, label='CTM-Inspired')
    if transformer_auroc is not None:
        plt.axhline(y=transformer_auroc, linestyle='--', color='gray',
                     label=f'Transformer ({transformer_auroc:.4f})')
    plt.xlabel('Internal Ticks (Compute)')
    plt.ylabel('Test AUROC')
    plt.title('Anytime Prediction: AUROC vs Internal Ticks')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'anytime_curve.png', dpi=150)
    plt.close()
    print("  Saved anytime_curve.png")


def plot_flip_rates():
    """Bar chart comparing prediction flip rates across all models."""
    data = load_json('main_results.json')
    if data is None:
        return

    models = []
    flip_rates = []
    for mtype in ['logistic', 'lstm', 'transformer', 'ctm_inspired', 'ctm_full']:
        if mtype in data and 'flip_rate' in data[mtype]:
            models.append(mtype.replace('_', ' ').title())
            flip_rates.append(data[mtype]['flip_rate'])

    if not models:
        print("  No flip rate data found.")
        return

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette('viridis', len(models))
    plt.bar(models, flip_rates, color=colors)
    plt.ylabel('Prediction Flip Rate')
    plt.title('Belief Stability: Prediction Flip Rates')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'flip_rates.png', dpi=150)
    plt.close()
    print("  Saved flip_rates.png")


def plot_model_comparison():
    """Grouped bar chart: accuracy and AUROC for all models."""
    data = load_json('main_results.json')
    if data is None:
        return

    models = []
    accs = []
    aurocs = []
    for mtype in ['logistic', 'lstm', 'transformer', 'ctm_inspired', 'ctm_full']:
        if mtype in data and 'accuracy' in data[mtype]:
            models.append(mtype.replace('_', ' ').title())
            accs.append(data[mtype]['accuracy'])
            aurocs.append(data[mtype]['auroc'])

    if not models:
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, label='Accuracy')
    ax.bar(x + width / 2, aurocs, width, label='AUROC')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Accuracy and AUROC')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'model_comparison.png', dpi=150)
    plt.close()
    print("  Saved model_comparison.png")


def plot_shock_recovery():
    """Plot shock accuracy and recovery steps for each model under each shock type."""
    data = load_json('shock_results.json')
    if data is None:
        return

    for shock_type, model_results in data.items():
        models = []
        shock_accs = []
        recovery_steps = []

        for mtype in ['logistic', 'lstm', 'transformer', 'ctm_inspired', 'ctm_full']:
            if mtype in model_results and 'shock_accuracy' in model_results[mtype]:
                models.append(mtype.replace('_', ' ').title())
                shock_accs.append(model_results[mtype]['shock_accuracy'])
                recovery_steps.append(model_results[mtype]['recovery_steps'])

        if not models:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Shock accuracy
        colors = sns.color_palette('viridis', len(models))
        ax1.bar(models, shock_accs, color=colors)
        ax1.set_ylabel('Accuracy During Shock')
        ax1.set_title(f'{shock_type.title()} Shock: Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=15)

        # Recovery steps
        ax2.bar(models, recovery_steps, color=colors)
        ax2.set_ylabel('Recovery Steps')
        ax2.set_title(f'{shock_type.title()} Shock: Recovery Time')
        ax2.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'shock_recovery_{shock_type}.png', dpi=150)
        plt.close()
        print(f"  Saved shock_recovery_{shock_type}.png")


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    print("Generating figures from experiment results...")

    plot_anytime_curve()
    plot_flip_rates()
    plot_model_comparison()
    plot_shock_recovery()

    print(f"Done. Figures saved in {PLOTS_DIR}")


if __name__ == "__main__":
    main()

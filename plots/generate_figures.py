import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def plot_anytime_curve():
    """
    Plots AUROC vs number of internal CTM ticks.
    """
    # Expected results from project design
    ticks = [1, 2, 3, 5, 10]
    ctm_auroc = [0.55, 0.56, 0.57, 0.58, 0.582]
    transformer_auroc = [0.575] * 5
    
    plt.figure(figsize=(8, 5))
    plt.plot(ticks, ctm_auroc, marker='o', label='CTM-Inspired')
    plt.plot(ticks, transformer_auroc, linestyle='--', color='gray', label='Transformer (Fixed Compute)')
    plt.xlabel('Internal Ticks (Compute)')
    plt.ylabel('Test AUROC')
    plt.title('Anytime Prediction Curve Expected Results')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/anytime_curve.png')
    plt.close()

def plot_flip_rates():
    """
    Bar chart comparing prediction flip rates across models.
    """
    # Expected results from project design
    models = ['Logistic', 'LSTM', 'Transformer', 'CTM-Inspired']
    flip_rates = [0.35, 0.28, 0.25, 0.15]
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=flip_rates, palette='viridis')
    plt.ylabel('Prediction Flip Rate')
    plt.title('Belief Stability Expected Results')
    plt.savefig('plots/flip_rates.png')
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent.parent
    plots_dir = base_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    plot_anytime_curve()
    plot_flip_rates()
    print(f"Figures successfully generated in {plots_dir}")

if __name__ == "__main__":
    main()

import numpy as np
from sklearn.metrics import roc_auc_score

def get_accuracy(predictions, labels):
    """
    Standard accuracy calculation.
    """
    return (predictions == labels).mean()

def get_auroc(logits, labels):
    """
    Computes Area Under ROC for binary classification.
    """
    # Assuming sigmoid/softmax has been applied or raw logits
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return roc_auc_score(labels, probs[:, 1])

def get_flip_rate(predictions):
    """
    Measures the stability of predictions over time.
    High flip rate indicates 'belief jitter'.
    """
    if len(predictions) < 2:
        return 0.0
    flips = (predictions[1:] != predictions[:-1]).sum()
    return flips / len(predictions)

def calculate_reaction_latency(predictions, labels, shock_indices):
    """
    Measures how many ticks it takes for the model to recover 
    and produce correct predictions after a synthetic shock.
    """
    latencies = []
    for start_idx in shock_indices:
        # Check how many steps after shock until prediction matches label consistently
        # (Simplified implementation)
        for t in range(start_idx, len(predictions)):
            if predictions[t] == labels[t]:
                latencies.append(t - start_idx)
                break
    return np.mean(latencies) if latencies else -1

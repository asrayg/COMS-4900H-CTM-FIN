import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_model(model, sim, device='cpu'):
    """Full streaming evaluation tracking accuracy, AUROC, and flip rate"""
    model.eval()
    sim.reset()
    state = model.reset_state(batch_size=1)
    
    all_logits = []
    all_y = []
    
    with torch.no_grad():
        while True:
            x_t, y_t = sim.step()
            if x_t is None:
                break
                
            x_t_tensor = torch.FloatTensor(x_t).unsqueeze(0).to(device)
            logits, state = model.forward_step(x_t_tensor, state)
            
            # Since forward_step might return multiple ticks, grab the final prediction
            if isinstance(logits, list):
                logits = logits[-1]
                
            all_logits.append(logits.cpu().numpy().item() if logits.numel() == 1 else logits.cpu().numpy()[0])
            all_y.append(y_t)
            
    if not all_logits:
        return {'accuracy': 0, 'auroc': 0.5, 'flip_rate': 0}
        
    all_logits = np.array(all_logits)
    all_y = np.array(all_y)
    
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs > 0.5).astype(int)
    
    acc = accuracy_score(all_y, preds)
    
    # Handle single class cases
    try:
        auroc = roc_auc_score(all_y, probs)
    except ValueError:
        auroc = 0.5
        
    flips = np.sum(preds[1:] != preds[:-1]) / len(preds) if len(preds) > 1 else 0.0
    
    return {
        'accuracy': acc,
        'auroc': auroc,
        'flip_rate': flips
    }

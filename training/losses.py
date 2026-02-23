import torch
import torch.nn as nn

class CTMDualLoss(nn.Module):
    """
    Implements the min-loss + max-certainty objective described in Sakana AI paper.
    """
    def __init__(self, criterion=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.criterion = criterion
        
    def forward(self, logits_list, target):
        """
        logits_list: list of logits from internal ticks.
        target: true label.
        """
        losses = []
        certainties = []
        for logits in logits_list:
            loss = self.criterion(logits, target)
            losses.append(loss)
            
            # Certainty estimation:
            prob = torch.sigmoid(logits)
            # Binary entropy style certainty (distance from 0.5)
            certainty = torch.abs(prob - 0.5) * 2.0 
            certainties.append(certainty)
            
        losses = torch.stack(losses)
        certainties = torch.stack(certainties)
        
        # Min loss over all ticks
        min_loss_idx = torch.argmin(losses)
        min_loss = losses[min_loss_idx]
        
        # Max certainty over all ticks
        max_cert_idx = torch.argmax(certainties)
        max_cert_loss = losses[max_cert_idx]
        
        # Dual objective (equally weighted in standard implementation)
        return (min_loss + max_cert_loss) / 2.0

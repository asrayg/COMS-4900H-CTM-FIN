import torch
import torch.nn as nn
from tqdm import tqdm
from training.losses import CTMDualLoss

class StreamingTrainer:
    def __init__(self, model, train_sim, val_sim, optimizer, criterion, device='cpu',
                 use_dual_loss=False):
        self.model = model.to(device)
        self.train_sim = train_sim
        self.val_sim = val_sim
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_dual_loss = use_dual_loss
        if use_dual_loss:
            self.dual_loss = CTMDualLoss(criterion=criterion)

    def train_epoch(self):
        self.model.train()
        self.train_sim.reset()
        state = self.model.reset_state(batch_size=1)
        total_loss = 0
        steps = 0

        pbar = tqdm(total=len(self.train_sim.X), desc="  train", leave=False,
                    unit="step", dynamic_ncols=True)

        while True:
            x_t, y_t = self.train_sim.step()
            if x_t is None:
                break

            x_t_tensor = torch.FloatTensor(x_t).unsqueeze(0).to(self.device)
            y_t_tensor = torch.FloatTensor([y_t]).to(self.device)

            if self.use_dual_loss:
                logits_list, state = self.model.forward_step(x_t_tensor, state,
                                                              return_all_ticks=True)
                loss = self.dual_loss(logits_list, y_t_tensor)
            else:
                logits, state = self.model.forward_step(x_t_tensor, state)
                if isinstance(logits, list):
                    logits = logits[-1]
                loss = self.criterion(logits, y_t_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            pbar.update(1)
            if steps % 500 == 0:
                pbar.set_postfix(loss=f"{total_loss / steps:.4f}")

            # Truncated BPTT state detachment
            if state is not None:
                if isinstance(state, tuple):
                    state = tuple(s.detach() if isinstance(s, torch.Tensor) else s for s in state)
                elif isinstance(state, torch.Tensor):
                    state = state.detach()

        pbar.close()
        return (total_loss / steps) if steps > 0 else 0

    def validate(self):
        self.model.eval()
        self.val_sim.reset()
        state = self.model.reset_state(batch_size=1)
        correct = 0
        total = 0

        with torch.no_grad():
            while True:
                x_t, y_t = self.val_sim.step()
                if x_t is None:
                    break
                x_t_tensor = torch.FloatTensor(x_t).unsqueeze(0).to(self.device)

                logits, state = self.model.forward_step(x_t_tensor, state)
                if isinstance(logits, list):
                    logits = logits[-1]
                preds = (logits > 0).float().squeeze()

                if preds.item() == y_t:
                    correct += 1
                total += 1

        return (correct / total) if total > 0 else 0

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ctm import ContinuousThoughtMachine
from data.loader import BinanceOrderBookDataset

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        # Initialize CTM states
        batch_size = x.shape[0]
        num_neurons = model.num_neurons
        z_prev = torch.zeros(batch_size, num_neurons).to(device)
        hist_pre_act = torch.zeros(batch_size, num_neurons, model.history_len).to(device)
        hist_post_act = torch.zeros(batch_size, num_neurons, 1).to(device) # Start with 1 tick
        
        optimizer.zero_grad()
        
        # CTM processes the sequence (x contains windows or streaming steps)
        # Simplified: process one batch of windows
        logits, _, _, _ = model(x[:, -1, :], z_prev, hist_pre_act, hist_post_act)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(loader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparams
    num_neurons = 64
    history_len = 16
    feature_dim = 50 # Example dim
    out_dim = 2 # Up/Down
    
    model = ContinuousThoughtMachine(num_neurons, history_len, feature_dim, out_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Dataset
    # train_ds = BinanceOrderBookDataset("data/binance_btc_l2.csv")
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=False) # Shuffle=False for temporal
    
    print("Codebase initialized. Ready for training.")

if __name__ == "__main__":
    main()

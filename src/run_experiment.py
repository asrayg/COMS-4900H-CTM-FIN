import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from src.models.ctm import ContinuousThoughtMachine
from src.models.baselines import TransformerBaseline, GRUBaseline
from src.data.loader import BinanceOrderBookDataset
from src.utils.metrics import get_auroc, get_flip_rate

def train_model(model, train_loader, val_loader, device, epochs=5, is_ctm=False):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_aurocs = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            if is_ctm:
                # CTM streaming state init
                batch_size = x.shape[0]
                z_prev = torch.zeros(batch_size, model.num_neurons).to(device)
                h_pre = torch.zeros(batch_size, model.num_neurons, model.history_len).to(device)
                h_post = torch.zeros(batch_size, model.num_neurons, 1).to(device)
                
                # Streaming through the window
                for t in range(x.shape[1]):
                    logits, z_prev, h_pre, h_post = model(x[:, t, :], z_prev, h_pre, h_post, ticks=1)
            else:
                logits = model(x)
                
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Validation
        val_auroc = evaluate(model, val_loader, device, is_ctm)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Val AUROC: {val_auroc:.4f}")
        val_aurocs.append(val_auroc)
        
    return val_aurocs

def evaluate(model, loader, device, is_ctm=False):
    model.eval()
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if is_ctm:
                batch_size = x.shape[0]
                z_prev = torch.zeros(batch_size, model.num_neurons).to(device)
                h_pre = torch.zeros(batch_size, model.num_neurons, model.history_len).to(device)
                h_post = torch.zeros(batch_size, model.num_neurons, 1).to(device)
                for t in range(x.shape[1]):
                    logits, z_prev, h_pre, h_post = model(x[:, t, :], z_prev, h_pre, h_post, ticks=3)
            else:
                logits = model(x)
            
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return get_auroc(all_logits, all_labels)

def run_experiment():
    device = torch.device("cpu") # Use CPU for stability in this env
    
    # Load Data
    train_ds = BinanceOrderBookDataset("data/train.csv", window_size=20)
    val_ds = BinanceOrderBookDataset("data/val.csv", window_size=20)
    test_ds = BinanceOrderBookDataset("data/test.csv", window_size=20)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    feature_dim = train_ds.features.shape[1]
    
    # Init Models
    ctm = ContinuousThoughtMachine(num_neurons=32, history_len=10, feature_dim=feature_dim, out_dim=2)
    transformer = TransformerBaseline(feature_dim=feature_dim, num_heads=4, num_layers=2, window_size=20, out_dim=2)
    
    print("--- Training CTM ---")
    ctm_aurocs = train_model(ctm, train_loader, val_loader, device, epochs=3, is_ctm=True)
    
    print("\n--- Training Transformer ---")
    trans_aurocs = train_model(transformer, train_loader, val_loader, device, epochs=3, is_ctm=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(ctm_aurocs, label='CTM AUROC', marker='o')
    plt.plot(trans_aurocs, label='Transformer AUROC', marker='s')
    plt.title("Convergence Comparison: CTM vs Transformer")
    plt.xlabel("Epoch")
    plt.ylabel("Validation AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig("results_convergence.png")
    print("\nConvergence plot saved as results_convergence.png")

if __name__ == "__main__":
    run_experiment()

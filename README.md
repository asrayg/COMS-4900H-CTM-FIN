# COMS-4900H-CTM-FIN: Continuous Thought Machines for Financial Microstructure

This repository contains the implementation of Continuous Thought Machines (CTMs) for high-frequency Bitcoin order book prediction.

## Repository Structure
- `src/models/`: Implementation of Neuron-Level Models (NLMs), Synchronization layers, and CTM architectures.
- `src/data/`: Data ingestion, clock-time resampling, and feature engineering for Binance L2 data.
- `src/utils/`: Helper functions for metrics (AUROC, Flip Rate, etc.) and visualization.
- `notebooks/`: Exploration and result visualization.
- `scripts/`: Training and evaluation scripts.

## Core Architecture
The system uses the **Synchronization Representation** as its primary latent state, maintaining a continuous reasoning flow through "internal ticks" decoupled from input clock-time.

## Baseline Comparisons
- Transformer (Sliding Window)
- GRU (Streaming Recurrence)
- Simple Logistic Regression

## Setup
```bash
# Recommended environment setup
conda create -n ctm-fin python=3.10
conda activate ctm-fin
pip install torch pandas numpy matplotlib binance-connector
```

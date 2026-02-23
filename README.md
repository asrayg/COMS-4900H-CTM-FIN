# Continuous Thought Machines for Streaming Reasoning in Financial Microstructure

This repository contains the codebase and experiment tracking for a research project on evaluating Continuous Thought Machines (CTMs) against traditional episodic models (e.g., Transformers) and recurrent models (e.g., LSTMs) in the domain of high-frequency Bitcoin order book microstructure prediction.

## Repository Layout
- `data/`: Fetches, pre-processes, and streams raw Binance order book data.
- `models/`: Implementations of the baseline models and CTM architectures.
- `training/`: Training routines and evaluation metrics.
- `configs/`: YAML configurations for training and parameter ablation studies.
- `experiments/`: Scripts to launch batch runs and analyze results.
- `shock_tests/`: Injects synthetic market shocks and assesses robustness.
- `plots/`: Produces visual figures for the academic paper and poster.

## Getting Started

1. Create a Python virtual environment and activate it:
```bash
python -m venv ctm_env
source ctm_env/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the data utilizing the Binance API tool. You have two options:
**Option A: Historical Data Downloads (Recommended for Large Datasets)**
```bash
python data/download_binance.py --mode historical --start 2024-03-01 --end 2024-03-05
```

**Option B: Live WebSocket Orderbook Tracking**
```bash
python data/download_binance.py --mode live --duration 60
```

4. Run the data preprocessing pipeline to build the unified `HDF5` dataset:
```bash
python data/preprocess.py
```

5. Run a standard experiment utilizing the configuration at `configs/default.yaml`:
```bash
python experiments/run_experiment.py
```

6. To run a sweep over hyper-parameter ablations:
```bash
python experiments/sweep_ablations.py
```

7. To evaluate the robustness of models under simulated market shocks:
```bash
python shock_tests/run_shock_eval.py --shock_type spread
```

8. Generate the paper figures:
```bash
python plots/generate_figures.py
```

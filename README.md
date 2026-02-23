# Continuous Thought Machines for Streaming Reasoning in Financial Microstructure

This repository contains the codebase and experiment tracking for a research project on evaluating Continuous Thought Machines (CTMs) against traditional episodic models (e.g., Transformers) and recurrent models (e.g., LSTMs) in the domain of high-frequency Bitcoin order book microstructure prediction.

## Repository Layout
- `data/`: Pre-processes and streams raw Tardis order book data.
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

3. Rebuild the orderbook snapshots from a Tardis CSV file:
```bash
python data/tardis_builder.py path/to/tardis_file.csv
```
This produces a parquet file in `data/raw/`.

4. Build the unified HDF5 dataset from the parquet files:
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

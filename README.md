# FlowSculptor

FlowSculptor is a spatio-temporal modeling project for traffic flow forecasting, with a focus on improving prediction performance in high-flow scenarios.

## Project Overview

The project includes:

- Temporal convolution-based feature extraction modules
- Learnable graph propagation structures
- Auxiliary branches and loss designs for high-flow regions
- Training and evaluation scripts for datasets such as PEMS and Manchester

## Project Structure

```text
FlowSculptor/
  configs/           # Training and evaluation configuration files
  data/              # Dataset directory
  experiments/       # Experimental scripts or results
  flowsculptor/      # Core model implementation
    data.py
    engine.py
    losses.py
    metrics.py
    model.py
  scripts/           # Training and evaluation entry points
    train.py
    evaluate.py
    plot_stdn_scalability.py
    prepare_scalability_data.py
    summarize_scalability.py
```

## Quick Start

1. Install dependencies

```bash
pip install torch numpy pandas scipy
```

2. Prepare data

Place dataset files under the data/ directory, for example:

- data/PEMS04/
- data/PEMS08/
- data/Manchester/

3. Train the model

```bash
python scripts/train.py --config configs/pems04.json
```


## Output Results

Training outputs will be saved to the run directory, for example:

```text
runs/PEMS04/
  best_model.pth
  config_used.json
  train.log
```

## Notes

- You can adjust datasets, model parameters, and training hyperparameters through the JSON files in configs/.
- If you want to run more experiments, you can continue extending the contents under experiments/.


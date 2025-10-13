# Custom Model Template for NEBULA Ecosystem

Quick guide for integrating your model with NEBULA Ecosystem.

## Required Modifications

### 1. `model_simulation.py`

Replace these three functions with your model's implementation:

```python
def create_model_policy(config):
    # Load your model here
    raise NotImplementedError("create_model_policy not implemented")

def nebula_to_model_obs(nebula_obs, task_description, info=None):
    # Convert NEBULA obs to your model's input format
    raise NotImplementedError("nebula_to_model_obs not implemented")

def model_to_nebula_action(model_action):
    # Convert your model's output to NEBULA action format (list of 8D arrays)
    raise NotImplementedError("model_to_nebula_action not implemented")
```

Update model-specific imports at the top.

### 2. `config.yaml`

Fill in these paths:

```yaml
experiment:
  dataset_root: "/path/to/dataset"
  save_dir: "./results"
  
model:
  model_path: "/path/to/checkpoint"
  # Add model-specific configs here

video:
  save_dir: "./videos"

figure:
  model_name: "YourModelName"
  save_dir: "./figures"
```

### 3. `run_nebula_test.sh`

Update `MODEL_NAME` variable at the top.

## Usage

```bash
cd baselines/custom

# Run all tests
bash run_nebula_test.sh

# Capability tests only
python model_simulation.py --config config.yaml --test-type capability

# Stress tests only
python model_simulation.py --config config.yaml --test-type stress

# Generate capability chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "YourModelName" \
    --results-dir /path/to/json/file \
    --type capability

# Generate stress test chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "YourModelName" \
    --results-dir /path/to/json/file \
    --type stress
```
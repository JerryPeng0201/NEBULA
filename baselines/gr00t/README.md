# GR00T-1.5 Adapter

1. ## Setup

Clone Isaac [GR00T](https://github.com/NVIDIA/Isaac-GR00T) repository and replace the data configuration file:
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T.git baselines/gr00t/Isaac-GR00T
cp baselines/gr00t/data_config.py baselines/gr00t/Isaac-GR00T/gr00t/experiment/data_config.py
cp baselines/gr00t/nebula_dataset.py baselines/gr00t/Isaac-GR00T/gr00t/data/
```

Then, setup the environment following the Isaac GR00T repository.

2. ## Fine-tuning

Update `baselines/gr00t/config.yaml` with your nebula dataset path:
```yaml
experiment:
  dataset_root: "/path/to/your/nebula/dataset"
```

Then, begin fine-tuning with:

```bash
cd baselines/gr00t
python gr00t_finetune.py
```

(read [fine-tuning script](baselines/gr00t/gr00t_finetune.py) for more configuration)

3. ## Inference & Visualization

Update `baselines/gr00t/config.yaml` with your model checkpoint path:

```yaml
model:
  model_path: "/path/to/your/gr00t/checkpoint"
```

(read [config](baselines/gr00t/config.yaml) file for more configuration)

⭐️ **<span style="color:orange;">Run all tests with visualization:**</span>

```bash
cd baselines/gr00t
bash run_nebula_test.sh
```

Or run tests/visualization separately:

```bash
cd baselines/gr00t

# Capability tests only
python gr00t_simulation.py --config config.yaml --test-type capability

# Stress tests only
python gr00t_simulation.py --config config.yaml --test-type stress

# Generate capability chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "GR00T-1.5" \
    --results-dir /path/to/gr00t_json_results \
    --type capability

# Generate stress test chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "GR00T-1.5" \
    --results-dir /path/to/gr00t_json_results \
    --type stress
```

**Note:** Make sure to install the required dependencies for each model according to their respective documentation. Most models require specific versions of PyTorch, CUDA, and other libraries.
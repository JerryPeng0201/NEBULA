# SpatialVLA Adapter

1. ## Setup

Clone the [SpatialVLA](https://github.com/SpatialVLA/SpatialVLA) repository and replace the dataset file:

```bash
# Clone the repository
cd baselines/SpatialVLA
git clone https://github.com/SpatialVLA/SpatialVLA.git

# copy the dataset file
cp nebula_dataset_integration.py SpatialVLA/data/
cp nebula_dataset.py SpatialVLA/data/
```

Then, setup the environment following the SpatialVLA repository.

2. ## Fine-tuning

First, Update `baselines/SpatialVLA/nebula_finetune_cfg.json` with your nebula dataset path:
```json
{
"nebula_data_root": "/path/to/your/nebula/dataset"
}
```

Then, begin fine-tuning with:

```bash
cd baselines/SpatialVLA
bash nebula_finetune.sh
```

3. ## Inference & Visualization

Update `baselines/SpatialVLA/config.yaml` with your model checkpoint path:

```yaml
model:
  model_path: "/path/to/your/SpatialVLA/checkpoint"
```

(read [config](baselines/SpatialVLA/config.yaml) file for more configuration)

⭐️ **<span style="color:orange;">Run all tests with visualization:**</span>

```bash
cd baselines/SpatialVLA
bash run_nebula_test.sh
```

Or run tests/visualization separately:

```bash
cd baselines/SpatialVLA

# Capability tests only
python spatialvla_simulation.py --config config.yaml --test-type capability

# Stress tests only
python spatialvla_simulation.py --config config.yaml --test-type stress

# Generate capability chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "SpatialVLA" \
    --results-dir path/to/spatialvla_json_results \
    --type capability

# Generate stress test chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "SpatialVLA" \
    --results-dir path/to/spatialvla_json_results \
    --type stress
```

**Note:** Make sure to install the required dependencies for each model according to their respective documentation. Most models require specific versions of PyTorch, CUDA, and other libraries.
# RDT-1B Adapter

1. ## Setup

Clone the [Robotics Diffusion Transformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) repository:

```bash
# Clone the repository
cd baselines/RDT
git clone https://github.com/thu-ml/RoboticsDiffusionTransformer.git

# Copy integration files (if needed)
# cp [nebula_integration_files] RoboticsDiffusionTransformer/
```
2. ## Fine-tuning

Begin fine-tuning with:

```bash
cd baselines/RDT
bash finetune_nebula.sh
```

3. ## Inference & Visualization

Update `baselines/RDT/config.yaml` with your model checkpoint path:

```yaml
model:
  model_path: "/path/to/your/rdt/checkpoint"
```

(read [config](baselines/RDT/config.yaml) file for more configuration)

⭐️ **<span style="color:orange;">Run all tests with visualization:**</span>

```bash
cd baselines/RDT
bash run_nebula_test.sh
```

Or run tests/visualization separately:

```bash
cd baselines/RDT

# Capability tests only
python rdt_simulation.py --config config.yaml --test-type capability

# Stress tests only
python rdt_simulation.py --config config.yaml --test-type stress

# Generate capability chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "RDT" \
    --results-dir /path/to/rdt_json_results \
    --type capability

# Generate stress test chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "RDT" \
    --results-dir /path/to/rdt_json_results \
    --type stress
```

**Note:** Make sure to install the required dependencies for each model according to their respective documentation. Most models require specific versions of PyTorch, CUDA, and other libraries.
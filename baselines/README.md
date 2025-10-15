# 🎯 Baseline Model Adapters

Standardized adapters for integrating Vision-Language-Action models with NEBULA.

## 📋 Available Baselines

| 🤖 Baseline | Fine-tuning | Inference & Visualization | Checkpoint |
|-------------|-------------|---------------------------|------------|
| [GR00T-1.5](/baselines/gr00t/) | ✅ | ✅ | 🚧 |
| [SpatialVLA](/baselines/SpatialVLA/) | ✅ | ✅ | 🚧 |
| [RDT-1B](/baselines/RDT/) | ✅ | ✅ | 🚧 |
| [Diffusion Policy](/baselines/DP/) | ✅ | ✅ | 🚧 |
| MT-ACT | 🚧 | 🚧 | 🚧 |
| ACT | 🚧 | 🚧 | 🚧 |

**TODO:** We are going to release more model adapters and their checkpoints soon.

## 🔧 Add Your Model

Use the **[Custom Template](./custom/README.md)** to integrate your own VLA model:

1. Implement three functions in `model_simulation.py`
2. Configure `config.yaml`
3. Run `bash run_nebula_test.sh`

## 📊 Compare Models

Generate graphs for multiple model comparsion:

```bash
python nebula/visualization/generate_graph_multiple.py --all
```

---

See individual model folders for detailed documentation.
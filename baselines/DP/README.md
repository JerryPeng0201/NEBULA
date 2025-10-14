# Diffusion Policy Adapter

1. ## Setup
Clone the modified version of [Diffusion Policy](https://github.com/LBG21/diffusion_policy) (orginal [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) repository):

```bash
# Clone the repository
cd baselines/DP
git clone https://github.com/LBG21/diffusion_policy.git
```
Then, setup the environment following the Diffusion Policy repository.

2. ## Fine-tuning

Download additional files for fine-tuning:

```bash
git clone --depth 1 https://github.com/haosulab/ManiSkill.git
mv ManiSkill/examples/baselines/diffusion_policy/diffusion_policy ./diffusion_policy
mv ManiSkill/mani_skill ./diffusion_policy
rm -rf ManiSkill
```

Then, begin fine-tuning with:

```bash
python nebula_train.py
```
(To change checkpoint save directory, edit the path in **save_ckp()** function) 

3. ## Inference & Visualization

Update `baselines/DP/config.yaml` with your model checkpoint path:

```yaml
model:
  model_path: "/path/to/your/DP/checkpoint"
```

(read [config](baselines/DP/config.yaml) file for more configuration)

⭐️ **<span style="color:orange;">Run all tests with visualization:**</span>

```bash
cd baselines/DP
bash run_nebula_test.sh
```

Or run tests/visualization separately:

```bash
cd baselines/DP

# Capability tests only
python dp_simulation.py --config config.yaml --test-type capability

# Stress tests only
python dp_simulation.py --config config.yaml --test-type stress

# Generate capability chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "Diffusion Policy" \
    --results-dir /path/to/dp_json_results \
    --type capability

# Generate stress test chart only
python ../../nebula/visualization/generate_graph_single.py \
    --model-name "Diffusion Policy" \
    --results-dir /path/to/dp_json_results \
    --type stress
```

**Note:** Make sure to install the required dependencies for each model according to their respective documentation. Most models require specific versions of PyTorch, CUDA, and other libraries.
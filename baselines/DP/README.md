# Diffusion Policy Adapter

1. ## Setup
Clone the [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) repository:

```bash
# Clone the repository
cd models/DP
git clone https://github.com/real-stanford/diffusion_policy
```
Then, setup the environment following the Diffusion Policy repository.

2. ## Fine-tuning

Begin finetuning with:

```bash
python nebula_train.sh
```
(To change checkpoint save directory, edit the path in **save_ckp()** function) 

3. ## Inference & Visualization

**Note:** Make sure to install the required dependencies for each model according to their respective documentation. Most models require specific versions of PyTorch, CUDA, and other libraries.
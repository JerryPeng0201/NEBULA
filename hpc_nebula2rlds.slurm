#!/bin/bash
#SBATCH --job-name=NEBULA2RLDS
#SBATCH -o aisc_a100_test.o
#SBATCH -A vxc204_aisc
#SBATCH -p aisc
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --exclude=aisct01,aisct04
#SBATCH --output=/mnt/rds/VipinRDS/VipinRDS/users/jxp1146/projects/logs/data_%j.out
#SBATCH --error=/mnt/rds/VipinRDS/VipinRDS/users/jxp1146/projects/logs/data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jxp1146@case.edu

source ~/.bashrc

module load CUDA/12.8.0

conda activate nebula

echo "CUDA loaded from: $CUDA_HOME"
nvcc --version
which python
python -c "import torch; print(torch.version.cuda, torch.__version__)"

export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
#export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
# /HDD1/embodied_ai/data/Nebula/Nebula-alpha
# /mnt/rds/VipinRDS/VipinRDS/users/jxp1146/data/nebula/alpha

vulkaninfo | head -n 40

cd /mnt/rds/VipinRDS/VipinRDS/users/jxp1146/projects/NEBULA

bash scripts/nebula2rlds.sh 
#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --account=torch_pr_1400_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
set -u
cd "$SLURM_SUBMIT_DIR"
PY=/scratch/zl7470/homework/project1/venv/bin/python
hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$PY src/train.py --arch res443 --epochs 3 --out runs/smoke_res443
$PY src/train.py --arch res443 --epochs 3 --compile --out runs/smoke_res443_c
$PY src/train.py --arch wrn28_10 --epochs 2 --out runs/smoke_wrn28_10
$PY src/train.py --arch res443 --epochs 2 --teacher runs/smoke_wrn28_10/model.pt --out runs/smoke_kd

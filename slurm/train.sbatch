#!/bin/bash
# Generic single-GPU training job. All arguments are forwarded to src/train.py, e.g.
#   sbatch -J res443 slurm/train.sbatch --arch res443 --epochs 200 --out runs/res443
#SBATCH --job-name=cifar
#SBATCH --account=torch_pr_1400_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --requeue

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PY=${PY:-/scratch/zl7470/homework/project1/venv/bin/python}
echo "host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
srun --unbuffered "$PY" src/train.py "$@"
echo "end=$(date)"

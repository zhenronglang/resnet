#!/bin/bash
# Run an arbitrary short command on one GPU, e.g.
#   sbatch slurm/run.sbatch python src/eval_ensemble.py --ckpts ... --out results/teacher_ensemble.json
#SBATCH --job-name=run
#SBATCH --account=torch_pr_1400_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PATH=/scratch/zl7470/homework/project1/venv/bin:$PATH
"$@"

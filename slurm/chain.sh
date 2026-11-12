#!/bin/bash
# Submit one training run as a chain of short Slurm jobs (short jobs get backfilled much sooner on a busy cluster).
# Each job resumes from <out>/ckpt_last.pt; once final.json exists the remaining jobs exit immediately.
#   bash slurm/chain.sh <name> <n_chunks> [--dep <jobid[:jobid]>] -- <train.py args...>
# Prints the job id of the last chunk.
set -euo pipefail
name=$1; n=$2; shift 2
dep=""
if [ "${1:-}" = "--dep" ]; then dep=$2; shift 2; fi
[ "${1:-}" = "--" ] && shift
prev=$dep
for i in $(seq 1 "$n"); do
  opt=(-J "$name" --time=02:00:00 --cpus-per-task=8 --mem=32G --parsable)
  [ -n "$prev" ] && opt+=(--dependency=afterany:$prev)
  prev=$(sbatch "${opt[@]}" slurm/train.sbatch "$@")
done
echo "$prev"

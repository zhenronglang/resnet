#!/bin/bash
# Third round.
# Stage P: PyramidNet-110 (alpha=270) + ShakeDrop teachers; stage F: students distilled from
# larger / more diverse teacher ensembles.
set -euo pipefail
sb() { local name=$1; shift; sbatch -J "$name" slurm/train.sbatch "$@"; }
PYR="--arch pyramid110_270 --epochs 600 --label-smoothing 0 --wd 1e-4"
sb P_split_s0 $PYR --seed 0 --out runs/P_pyr110_split_s0
for s in 0 1 2; do
  sb P_full_s$s $PYR --seed $s --val-size 0 --out runs/P_pyr110_full_s$s
done
# students from the six existing WRN-28-10 teachers (4 plain + 2 SAM)
T6S="runs/T_wrn28_10_split_s0/model.pt runs/T_wrn28_10_split_s1/model.pt runs/T_wrn28_10_split_s2/model.pt runs/T_wrn28_10_split_s3/model.pt runs/T_wrn28_10_sam_split_s0/model.pt runs/T_wrn28_10_sam_split_s1/model.pt"
T6F="runs/T_wrn28_10_full_s0/model.pt runs/T_wrn28_10_full_s1/model.pt runs/T_wrn28_10_full_s2/model.pt runs/T_wrn28_10_full_s3/model.pt runs/T_wrn28_10_sam_full_s0/model.pt runs/T_wrn28_10_sam_full_s1/model.pt"
sb F_kd6t --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 --teacher $T6S --out runs/F_res443_kd6t_e800
sb F_kd6t_full --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 --val-size 0 --teacher $T6F --out runs/F_res443_kd6t_e800_full

# ---- G/H: students distilled from PyramidNet teachers (+ WRN teachers); start when the teachers are done.
# Usage: bash slurm/overnight.sh is for stage P/F; the lines below were submitted with --dependency on the P jobs.
# P_SPLIT=<jobid>; P_FULL=<id0>:<id1>:<id2>
# G: PyramidNet + WRN teachers, each family gets half of the total weight
# sbatch --dependency=afterok:$P_SPLIT -J G_kdmix slurm/train.sbatch --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 \
#   --teacher runs/P_pyr110_split_s0/model.pt $T6S --teacher-weights 6 1 1 1 1 1 1 --out runs/G_res443_kdmix_e800
# sbatch --dependency=afterok:$P_FULL -J G_kdmix_full slurm/train.sbatch --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 --val-size 0 \
#   --teacher runs/P_pyr110_full_s{0,1,2}/model.pt $T6F --teacher-weights 2 2 2 1 1 1 1 1 1 --out runs/G_res443_kdmix_e800_full
# sbatch --dependency=afterok:$P_FULL -J H_kdpyr_full slurm/train.sbatch --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 --val-size 0 \
#   --teacher runs/P_pyr110_full_s{0,1,2}/model.pt --out runs/H_res443_kdpyr_e800_full

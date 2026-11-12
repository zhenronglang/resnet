#!/bin/bash
# Every experiment reported in the notebook / report, in order. Run from the repository root.
# Each line submits one single-GPU job; without Slurm replace "$SB" with "python src/train.py".
# Stages B/C need the teachers of stage T to be finished (use --dependency=afterok:<jobid> if desired).
set -euo pipefail
SB="sbatch slurm/train.sbatch"

# ---- A: architecture / recipe screening on the 45k/5k split (no distillation) ----
for a in res443 wrn22_4 res443_post rn18_w40 post643 post_w80; do
  $SB --arch $a --epochs 200 --out runs/A_${a}_e200
done
$SB --arch res443_post --epochs 200 --seed 1 --out runs/A_res443_post_e200_s1
$SB --arch res443_post --epochs 200 --mix cutmix --out runs/A_res443_post_cutmix_e200
$SB --arch res443 --epochs 400 --out runs/A_res443_e400
$SB --arch res443_post --epochs 400 --out runs/A_res443_post_e400

# ---- T: WideResNet-28-10 teachers (36.5M params, used only during training), no label smoothing ----
for s in 0 1; do
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --out runs/T_wrn28_10_split_s$s
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --val-size 0 --out runs/T_wrn28_10_full_s$s
done

# ---- B: knowledge distillation on the 45k/5k split (teachers trained on the same 45k) ----
TS="runs/T_wrn28_10_split_s0/model.pt runs/T_wrn28_10_split_s1/model.pt"
$SB --arch res443_post --epochs 400 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_post_kd_a09T4_e400
$SB --arch res443_post --epochs 800 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_post_kd_a09T4_e800
$SB --arch res443_post --epochs 1600 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_post_kd_a09T4_e1600
$SB --arch res443_post --epochs 400 --teacher $TS --kd-alpha 1.0 --kd-temp 1 --mix cutmix --out runs/B_res443_post_kd_a1T1_cutmix_e400
$SB --arch res443_post --epochs 800 --teacher $TS --kd-alpha 1.0 --kd-temp 1 --mix cutmix --out runs/B_res443_post_kd_a1T1_cutmix_e800
$SB --arch res443_post --epochs 400 --teacher runs/T_wrn28_10_split_s0/model.pt --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_post_kd1teacher_a09T4_e400
$SB --arch post643 --epochs 400 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_post643_kd_a09T4_e400
$SB --arch res443 --epochs 400 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_kd_a09T4_e400
$SB --arch res443 --epochs 800 --teacher $TS --kd-alpha 0.9 --kd-temp 4 --out runs/B_res443_kd_a09T4_e800

# ---- C: final models on all 50k images (teachers trained on all 50k) ----
TF="runs/T_wrn28_10_full_s0/model.pt runs/T_wrn28_10_full_s1/model.pt"
$SB --arch res443_post --epochs 800 --val-size 0 --teacher $TF --kd-alpha 0.9 --kd-temp 4 --out runs/C_res443_post_kd_a09T4_e800_full
$SB --arch res443_post --epochs 800 --val-size 0 --teacher $TF --kd-alpha 1.0 --kd-temp 1 --mix cutmix --out runs/C_res443_post_kd_a1T1_cutmix_e800_full

$SB --arch res443 --epochs 400 --val-size 0 --teacher $TF --kd-alpha 0.9 --kd-temp 4 --out runs/C_res443_kd_a09T4_e400_full
$SB --arch res443 --epochs 800 --val-size 0 --teacher $TF --kd-alpha 0.9 --kd-temp 4 --out runs/C_res443_kd_a09T4_e800_full

# ---- evaluation helpers ----
# sbatch slurm/run.sbatch python src/eval_ensemble.py --ckpts $TS --out results/teacher_ensemble.json
# python src/collect_results.py
# python src/predict.py --ckpt runs/C_res443_kd_a09T4_e800_full/model.pt --out submission.csv   # first submission

# ---- second round (after the first Kaggle submission) ----
for s in 2 3; do
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --out runs/T_wrn28_10_split_s$s
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --val-size 0 --out runs/T_wrn28_10_full_s$s
done
for s in 0 1; do
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --sam 0.05 --seed $s --out runs/T_wrn28_10_sam_split_s$s
  $SB --arch wrn28_10 --epochs 300 --label-smoothing 0 --sam 0.05 --seed $s --val-size 0 --out runs/T_wrn28_10_sam_full_s$s
done
KD="--epochs 400 --teacher $TS --kd-alpha 0.9 --kd-temp 4"
$SB --arch res443 --upsample 48 $KD --out runs/D_res443_up48_kd_e400
$SB --arch res443 --upsample 64 $KD --out runs/D_res443_up64_kd_e400
$SB --arch res443 --sam 0.05 $KD --out runs/D_res443_sam05_kd_e400
$SB --arch pre_w80 $KD --out runs/D_pre_w80_kd_e400
$SB --arch post_w80 $KD --out runs/D_post_w80_kd_e400
$SB --arch res443 $KD --seed 1 --out runs/D_res443_kd_e400_s1
T4S="runs/T_wrn28_10_split_s0/model.pt runs/T_wrn28_10_split_s1/model.pt runs/T_wrn28_10_split_s2/model.pt runs/T_wrn28_10_split_s3/model.pt"
T4F="runs/T_wrn28_10_full_s0/model.pt runs/T_wrn28_10_full_s1/model.pt runs/T_wrn28_10_full_s2/model.pt runs/T_wrn28_10_full_s3/model.pt"
$SB --arch res443 --epochs 800 --teacher $T4S --kd-alpha 0.9 --kd-temp 4 --out runs/E_res443_kd4t_e800          # final, validation
$SB --arch res443 --epochs 800 --val-size 0 --teacher $T4F --kd-alpha 0.9 --kd-temp 4 --out runs/E_res443_kd4t_e800_full  # final
# sbatch slurm/run.sbatch python src/eval_ensemble.py --ckpts $T4S --out results/teacher_ensemble4.json
# sbatch slurm/run.sbatch python src/eval_tta.py --ckpts runs/B_res443_kd_a09T4_e800/model.pt --out results/tta_split.json
# python src/predict.py --ckpt runs/E_res443_kd4t_e800_full/model.pt --tta-shift 1 --out submission.csv   # final submission

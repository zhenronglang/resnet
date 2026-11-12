#!/bin/bash
# Same runs as slurm/overnight.sh, submitted as chains of 2-hour jobs (see slurm/chain.sh).
set -euo pipefail
C="bash slurm/chain.sh"
PYR="--arch pyramid110_270 --epochs 600 --label-smoothing 0 --wd 1e-4"
T6S="runs/T_wrn28_10_split_s0/model.pt runs/T_wrn28_10_split_s1/model.pt runs/T_wrn28_10_split_s2/model.pt runs/T_wrn28_10_split_s3/model.pt runs/T_wrn28_10_sam_split_s0/model.pt runs/T_wrn28_10_sam_split_s1/model.pt"
T6F="runs/T_wrn28_10_full_s0/model.pt runs/T_wrn28_10_full_s1/model.pt runs/T_wrn28_10_full_s2/model.pt runs/T_wrn28_10_full_s3/model.pt runs/T_wrn28_10_sam_full_s0/model.pt runs/T_wrn28_10_sam_full_s1/model.pt"
KD="--arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4"

ps=$($C P_split_s0 4 -- $PYR --seed 0 --out runs/P_pyr110_split_s0)
pf0=$($C P_full_s0 4 -- $PYR --seed 0 --val-size 0 --out runs/P_pyr110_full_s0)
pf1=$($C P_full_s1 4 -- $PYR --seed 1 --val-size 0 --out runs/P_pyr110_full_s1)
pf2=$($C P_full_s2 4 -- $PYR --seed 2 --val-size 0 --out runs/P_pyr110_full_s2)
$C F_kd6t 4 -- $KD --teacher $T6S --out runs/F_res443_kd6t_e800
$C F_kd6t_full 4 -- $KD --val-size 0 --teacher $T6F --out runs/F_res443_kd6t_e800_full
# students distilled from PyramidNet (+ WRN) teachers; each family gets half of the total teacher weight in G
$C G_kdmix 5 --dep $ps -- $KD --teacher runs/P_pyr110_split_s0/model.pt $T6S --teacher-weights 6 1 1 1 1 1 1 \
    --out runs/G_res443_kdmix_e800
$C G_kdmix_full 5 --dep $pf0:$pf1:$pf2 -- $KD --val-size 0 --teacher runs/P_pyr110_full_s{0,1,2}/model.pt $T6F \
    --teacher-weights 2 2 2 1 1 1 1 1 1 --out runs/G_res443_kdmix_e800_full
$C H_kdpyr_full 5 --dep $pf0:$pf1:$pf2 -- $KD --val-size 0 --teacher runs/P_pyr110_full_s{0,1,2}/model.pt \
    --out runs/H_res443_kdpyr_e800_full

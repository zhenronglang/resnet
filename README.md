# CIFAR-10 ResNet Challenge (≤ 5M parameters)

The goal is to design a residual network with **at most 5 million trainable parameters** that maximizes accuracy on a held-out version of CIFAR-10. Everything is trained from scratch on the 50,000 provided training images (no pre-trained weights, no extra data).

## Result

| | |
|---|---|
| Final model | `res443`: pre-activation ResNet, 3 stages × (4, 4, 3) basic blocks, 64 / 128 / 256 channels |
| Trainable parameters | **4,668,378** (limit 5,000,000) |
| Training | 800 epochs on all 50k images, knowledge distillation from four WideResNet-28-10 teachers |
| Inference | average over 10 views (image + horizontal flip, each also shifted by ±1 px) |
| Validation accuracy | **98.20%** (same recipe trained on 45k images, evaluated on the 5k held-out images) |
| Test accuracy (Kaggle, 2,000 test images) | **90.60%** |


The full experiment log, every training curve and the final evaluation are in the executed notebook
[`project1.ipynb`](project1.ipynb).

## Approach in short

1. **Architecture search under the budget (200 epochs, no distillation).** All candidates are ResNets built by
   `src/models.py`. Three-stage designs (32→16→8 resolution) with widths 64/128/256 use the budget best; a thin
   four-stage ResNet-18 (widths 40–320) was clearly worse (96.2% vs ~97.2%). Pre-activation (He et al., 2016b)
   and the original post-activation block from the assignment performed the same at 200 epochs.
2. **Training recipe.** SGD + Nesterov (lr 0.1, batch 128, wd 5e-4), 5 warm-up epochs + cosine decay,
   random crop + flip + TrivialAugment + Cutout(16), label smoothing 0.1, bf16 mixed precision, EMA of weights,
   horizontal-flip test-time augmentation. Doubling the schedule (200→400 epochs) gave +0.2–0.4%.
3. **Knowledge distillation.** The parameter limit applies only to the submitted model, so two WideResNet-28-10
   teachers (36.5M parameters, ~98.1% validation accuracy, trained without label smoothing) supervise the student
   on the same augmented batches (Hinton loss, α = 0.9, T = 4). This added another +0.4%; with distillation the
   pre-activation student was consistently better than the post-activation one.
4. **Schedule length.** With distillation, the 400- and 800-epoch pre-activation students are within noise on the
   validation split (98.04% vs 97.94%, 5 of 5,000 images); the 800-epoch recipe was used. No test labels are used
   anywhere.
5. **Second round.** The first submission scored 89.70% on Kaggle (97.94% validation): the test set is a natural,
   harder re-collection, so the remaining lever is in-distribution accuracy. Input upsampling to 48/64 px (same
   parameters, more compute), SAM, wider networks (`pre_w80`, `post_w80`) and SAM-trained teachers all stayed within
   seed noise (~0.15%). What did help: **four** teachers instead of two (97.94% → 98.18% at 800 epochs) and averaging
   predictions over the flip and ±1-pixel shifts (+0.1%).
6. **Final model.** That recipe was re-run on all 50k images (teachers also trained on all 50k) and produced
   `submission.csv`, which scored 90.60% on Kaggle.

Key validation numbers (5k held-out images, EMA weights + flip TTA):

| Run | Val acc |
|---|---|
| post-act `res443_post`, 200 epochs | 97.24% |
| … 400 epochs | 97.44% |
| … + distillation, 2 teachers (400 / 800 / 1600 epochs) | 97.86% / 97.96% / 97.82% |
| pre-act `res443`, 400 epochs | 97.64% |
| … + distillation, 2 teachers (400 epochs; second seed) | 98.04% (97.90%) |
| … + distillation, 2 teachers (800 epochs) — first submission, Kaggle 89.70% | 97.94% |
| … + inputs upsampled to 48 / 64 px (400 epochs) | 97.84% / 97.94% |
| … + SAM (400 epochs) | 97.88% |
| wider `pre_w80` / `post_w80` + distillation (400 epochs) | 98.04% / 98.10% |
| **pre-act `res443` + distillation, 4 teachers, 800 epochs — final recipe** | **98.18%** (98.20% with shift TTA) |
| teachers: WRN-28-10 (36.5M params each, not submitted) | 97.94–98.14% (ensembles: 2 → 98.18%, 4 → 98.28%, 6 → 98.34%) |

Note: the competition test images are not distributed like the CIFAR-10 training set (they are brighter and
higher-contrast; 2,000 images, consistent with a re-collected test set such as CIFAR-10.1), so the Kaggle accuracy
is expected to be a few points below the validation accuracy.

## Repository layout

| Path | Contents |
|---|---|
| `src/prepare_data.py` | converts the competition PNGs into `data/train.npz` / `data/test.npz` (one-time) |
| `src/models.py` | configurable ResNet (`ResNet`), named configurations, parameter counting |
| `src/train.py` | training: augmentation, SGD + cosine schedule, EMA, knowledge distillation, test-set prediction |
| `src/predict.py` | loads a checkpoint, prints #params (+ validation accuracy) and writes `submission.csv` |
| `src/eval_ensemble.py` | validation accuracy of the teachers and their ensemble |
| `src/eval_tta.py` | compares test-time-augmentation view sets |
| `src/export_checkpoint.py` | exports a slim checkpoint (EMA weights only) from a training run |
| `src/collect_results.py` | copies the log files of finished runs out of `runs/` for the notebook |
| `slurm/train.sbatch`, `slurm/run.sbatch` | single-GPU Slurm wrappers |
| `slurm/reproduce.sh` | every training job behind the reported results, in order |
| `project1.ipynb` | executed notebook: data, architecture summary, all training curves, final accuracy, submission |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch torchvision numpy pandas pillow matplotlib jupyter torchsummary
# place the Kaggle files (train_labels.csv, train_images/, test.csv, test_images/) in data/
python src/prepare_data.py --data-dir data
```

## Reproducing

All models are trained from scratch. `slurm/reproduce.sh` lists every training job behind the reported results, in
order (one single-GPU Slurm job per line; without Slurm run `python src/train.py <same arguments>`). The final model
needs the four teachers first, and `predict.py` then prints its parameter count and writes `submission.csv`:

```bash
for s in 0 1 2 3; do
  python src/train.py --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --val-size 0 --out runs/T_wrn28_10_full_s$s
done
python src/train.py --arch res443 --epochs 800 --val-size 0 --kd-alpha 0.9 --kd-temp 4 \
    --teacher runs/T_wrn28_10_full_s{0,1,2,3}/model.pt --out runs/E_res443_kd4t_e800_full
python src/predict.py --ckpt runs/E_res443_kd4t_e800_full/model.pt --tta-shift 1 --out submission.csv
```

The validation accuracy comes from the same recipe with 5,000 training images held out (the default `--val-size`):

```bash
for s in 0 1 2 3; do
  python src/train.py --arch wrn28_10 --epochs 300 --label-smoothing 0 --seed $s --out runs/T_wrn28_10_split_s$s
done
python src/train.py --arch res443 --epochs 800 --kd-alpha 0.9 --kd-temp 4 \
    --teacher runs/T_wrn28_10_split_s{0,1,2,3}/model.pt --out runs/E_res443_kd4t_e800
python src/predict.py --ckpt runs/E_res443_kd4t_e800/model.pt --tta-shift 1 --out /dev/null   # prints held-out val accuracy
```

Approximate cost on one NVIDIA A100/H200: a teacher ≈ 0.5–1 h, the final student ≈ 2.5–3 h (four teacher forward passes per step).

The notebook is saved with all its outputs. Re-executing it needs the logs and models of these runs: run
`python src/collect_results.py` and the evaluation commands commented in `slurm/reproduce.sh`, then export the two
final models with `src/export_checkpoint.py` to `checkpoints/final_model.pt` and `checkpoints/final_model_split.pt`.

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- K. He, X. Zhang, S. Ren, J. Sun. *Identity Mappings in Deep Residual Networks.* ECCV 2016.
- S. Zagoruyko, N. Komodakis. *Wide Residual Networks.* BMVC 2016.
- G. Hinton, O. Vinyals, J. Dean. *Distilling the Knowledge in a Neural Network.* 2015.
- L. Beyer et al. *Knowledge Distillation: A Good Teacher is Patient and Consistent.* CVPR 2022.
- R. Müller, S. Kornblith, G. Hinton. *When Does Label Smoothing Help?* NeurIPS 2019.
- S. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- T. DeVries, G. Taylor. *Improved Regularization of CNNs with Cutout.* 2017.
- S. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- P. Foret, A. Kleiner, H. Mobahi, B. Neyshabur. *Sharpness-Aware Minimization for Efficiently Improving Generalization.* ICLR 2021.
- B. Recht, R. Roelofs, L. Schmidt, V. Shankar. *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* 2018.
- Model/training-loop structure follows common practice from the `kuangliu/pytorch-cifar` repository (MIT license).

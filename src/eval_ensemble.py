"""Validation accuracy of individual checkpoints and of their softmax-averaged ensemble
(used to report the accuracy of the distillation teachers).

    python src/eval_ensemble.py --ckpts runs/T_a/model.pt runs/T_b/model.pt --out results/teacher_ensemble.json
"""
import argparse
import json
import os

import torch

from train import Normalizer, accuracy, load_split, load_teacher, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = Normalizer(dev)
    _, _, x_va, y_va = load_split(args.data_dir, args.val_size)
    res, probs = {"members": {}}, []
    for c in args.ckpts:
        m = load_teacher(c, dev)  # EMA weights
        z = predict_logits(m, x_va, norm, tta=True)
        probs.append(z.softmax(1))
        res["members"][c] = accuracy(z, y_va)
    res["ensemble"] = accuracy(torch.stack(probs).mean(0), y_va)
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

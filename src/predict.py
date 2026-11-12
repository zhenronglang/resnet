"""Generate a Kaggle submission from a trained checkpoint.

    python src/predict.py --ckpt runs/<run>/model.pt --out submission.csv [--no-tta] [--tta-shift 1] [--weights model]

Prints the parameter count (must be <= 5M) and, if the checkpoint was trained on the
45k/5k split, the held-out validation accuracy.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from models import build_model, count_params
from train import PARAM_LIMIT, Normalizer, accuracy, load_split, predict_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--tta-shift", type=int, default=0, help="also average over +-N pixel shifts (10 views)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], **ck.get("arch_overrides", {}))
    sd = ck["ema_state_dict"] if (args.weights == "ema" and ck.get("ema_state_dict")) else ck["state_dict"]
    model.load_state_dict(sd)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    n = count_params(model)
    print(f"arch: {ck['arch']}  trainable parameters: {n:,}  (limit {PARAM_LIMIT:,}, ok={n <= PARAM_LIMIT})")

    norm, tta = Normalizer(dev), not args.no_tta
    val_size = ck["args"].get("val_size", 0)
    if val_size:
        _, _, x_va, y_va = load_split(args.data_dir, val_size)
        print(f"held-out validation accuracy ({val_size} imgs, tta={tta}, shift={args.tta_shift}): "
              f"{100 * accuracy(predict_logits(model, x_va, norm, tta=tta, shift=args.tta_shift), y_va):.2f}%")

    test = np.load(os.path.join(args.data_dir, "test.npz"))
    x_te = torch.from_numpy(test["x"]).permute(0, 3, 1, 2).contiguous()
    z = predict_logits(model, x_te, norm, tta=tta, shift=args.tta_shift).cpu()
    pred = z.argmax(1).numpy()
    pd.DataFrame({"id": test["ids"], "label": pred}).to_csv(args.out, index=False)
    print(f"wrote {len(pred)} predictions to {args.out}; class histogram {np.bincount(pred, minlength=10)}")


if __name__ == "__main__":
    main()

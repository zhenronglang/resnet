"""Compare test-time augmentation (TTA) variants for trained checkpoints.

For each checkpoint trained on the 45k split: accuracy on the 5k held-out validation images for several TTA view
sets.

    python src/eval_tta.py --ckpts runs/B_x/model.pt ... --out results/tta.json
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

from train import Normalizer, accuracy, load_split, load_teacher

VIEWS = {
    "none": [(0, 0)],
    "flip": [(0, 0)],
    "flip+shift1": [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
    "flip+shift2": [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)],
    "flip+shift2diag": [(0, 0), (2, 2), (-2, 2), (2, -2), (-2, -2)],
    "flip+shift2all": [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (2, -2), (-2, -2)],
}


def shift(x, dx, dy, pad=4):
    if dx == 0 and dy == 0:
        return x
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    H, W = x.shape[2:]
    return xp[:, :, pad + dy:pad + dy + H, pad + dx:pad + dx + W]


@torch.no_grad()
def tta_logits(model, x_u8, norm, views, flip, bs=500):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x_u8), bs):
        xb = norm(x_u8[i:i + bs].to(dev)).contiguous()
        z = 0
        n = 0
        for dx, dy in views:
            v = shift(xb, dx, dy).contiguous(memory_format=torch.channels_last)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = z + model(v).float().log_softmax(1)
                n += 1
                if flip:
                    z = z + model(v.flip(3)).float().log_softmax(1)
                    n += 1
        out.append(z / n)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    norm = Normalizer(dev)
    _, _, x_va, y_va = load_split(args.data_dir, 5000)

    res = {}
    for c in args.ckpts:
        ck = torch.load(c, map_location="cpu", weights_only=False)
        if ck["args"].get("val_size", 5000) == 0:
            print(f"{c}: trained on all 50k images, no held-out validation images; skipped", flush=True)
            continue
        m = load_teacher(c, dev)
        res[c] = {}
        for name, views in VIEWS.items():
            flip = name != "none"
            r = {"val_acc": accuracy(tta_logits(m, x_va, norm, views, flip), y_va)}
            res[c][name] = r
            print(f"{c:55s} {name:16s} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

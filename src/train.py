"""Train a ResNet on the competition's CIFAR-10 data, optionally distilling from teacher(s).

Examples
--------
# plain training on a 45k/5k train/val split
python src/train.py --arch res443 --epochs 200 --val-size 5000 --out runs/res443_e200

# knowledge distillation from a trained teacher, full 50k training set
python src/train.py --arch res443 --epochs 600 --val-size 0 \
    --teacher runs/wrn28_10_full/model.pt --kd-alpha 1.0 --kd-temp 1.0 --out runs/res443_kd

Outputs in --out: args.json, log.jsonl (one line per epoch), model.pt (final weights + EMA weights),
test_logits.npz (TTA logits of the competition test set), submission.csv.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from models import build_model, count_params

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
PARAM_LIMIT = 5_000_000


# ----------------------------------------------------------------------------- data
class ArrayDataset(Dataset):
    """uint8 images (N, 3, 32, 32) + labels; augmentation runs in DataLoader workers."""

    def __init__(self, x, y, transform=None):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[i]


def make_train_transform(aug):
    t = [v2.RandomCrop(32, padding=4, padding_mode="reflect"), v2.RandomHorizontalFlip()]
    if aug == "trivial":
        t.append(v2.TrivialAugmentWide())
    elif aug == "autoaug":
        t.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10))
    elif aug == "randaug":
        t.append(v2.RandAugment(num_ops=2, magnitude=14))
    elif aug != "basic":
        raise ValueError(aug)
    return v2.Compose(t)


def load_split(data_dir, val_size, seed=0):
    d = np.load(os.path.join(data_dir, "train.npz"))
    x = torch.from_numpy(d["x"]).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(d["y"]).long()
    if val_size == 0:
        return x, y, None, None
    rng = np.random.RandomState(seed)  # stratified, fixed split shared by all experiments
    per_class = val_size // 10
    val_idx = np.concatenate([rng.permutation(np.where(d["y"] == c)[0])[:per_class] for c in range(10)])
    mask = np.ones(len(y), bool)
    mask[val_idx] = False
    tr = torch.from_numpy(np.where(mask)[0])
    va = torch.from_numpy(np.sort(val_idx))
    return x[tr], y[tr], x[va], y[va]


class Normalizer:
    def __init__(self, device):
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1) * 255
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1) * 255

    def __call__(self, x_u8):
        x = (x_u8.float() - self.mean) / self.std
        return x.contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- batch-level augmentation (GPU)
def cutout(x, size):
    """Zero a random size x size square per image (DeVries & Taylor 2017). x is normalized, so 0 == mean."""
    B, _, H, W = x.shape
    cy = torch.randint(0, H, (B, 1, 1), device=x.device)
    cx = torch.randint(0, W, (B, 1, 1), device=x.device)
    ys = torch.arange(H, device=x.device).view(1, H, 1)
    xs = torch.arange(W, device=x.device).view(1, 1, W)
    m = (ys >= cy - size // 2) & (ys < cy + size // 2) & (xs >= cx - size // 2) & (xs < cx + size // 2)
    return x * (~m).unsqueeze(1).to(x.dtype)


def mix_batch(x, y, mode, alpha):
    """mixup / cutmix. Returns mixed x, the permuted labels and lambda (fraction of the original image)."""
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        H, W = x.shape[2:]
        rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
        cy, cx = np.random.randint(H), np.random.randint(W)
        y0, y1 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
        x0, x1 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
        x = x.clone()
        x[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
        lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
    return x, y[perm], lam


# ----------------------------------------------------------------------------- EMA of weights
class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0
        self.src = [t for t in model.state_dict().values() if t.dtype.is_floating_point]
        self.dst = [t for t in self.model.state_dict().values() if t.dtype.is_floating_point]
        self.src_int = [t for t in model.state_dict().values() if not t.dtype.is_floating_point]
        self.dst_int = [t for t in self.model.state_dict().values() if not t.dtype.is_floating_point]

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_lerp_(self.dst, self.src, 1 - d)
        for a, b in zip(self.dst_int, self.src_int):
            a.copy_(b)


# ----------------------------------------------------------------------------- eval helpers
@torch.no_grad()
def predict_logits(model, x_u8, norm, tta=True, bs=1000, shift=0):
    """Logits for uint8 images. tta: average with the horizontal flip. shift > 0: additionally average over the
    image translated by +-shift pixels horizontally and vertically (reflect padding), each also flipped; with
    shift > 0 the log-probabilities of the 10 views are averaged."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x_u8), bs):
        xb = norm(x_u8[i:i + bs].to(dev, non_blocking=True))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            if shift > 0:
                H, W = xb.shape[2:]
                xp = F.pad(xb.contiguous(), (shift,) * 4, mode="reflect")
                z = 0
                for dx, dy in [(0, 0), (shift, 0), (-shift, 0), (0, shift), (0, -shift)]:
                    v = xp[:, :, shift + dy:shift + dy + H, shift + dx:shift + dx + W].contiguous(
                        memory_format=torch.channels_last)
                    z = z + model(v).float().log_softmax(1) + model(v.flip(3)).float().log_softmax(1)
                z = z / 10
            else:
                z = model(xb).float()
                if tta:
                    z = (z + model(xb.flip(3)).float()) / 2
        out.append(z)
    return torch.cat(out)


def accuracy(logits, y):
    return (logits.argmax(1).cpu() == y.cpu()).float().mean().item()


def load_teacher(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], **ck.get("arch_overrides", {}))
    m.load_state_dict(ck["ema_state_dict"] if ck.get("ema_state_dict") else ck["state_dict"])
    return m.to(device).to(memory_format=torch.channels_last).eval()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="res443")
    ap.add_argument("--drop", type=float, default=None, help="dropout inside residual blocks (overrides config)")
    ap.add_argument("--upsample", type=int, default=None, help="resize inputs to this resolution inside the model")
    ap.add_argument("--sam", type=float, default=0.0, help="SAM neighbourhood size rho (0 = plain SGD)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--val-size", type=int, default=5000, help="0 = train on all 50k images")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=5, help="warmup epochs")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug", default="trivial", choices=["basic", "trivial", "autoaug", "randaug"])
    ap.add_argument("--cutout", type=int, default=16, help="cutout square size (0 = off)")
    ap.add_argument("--mix", default="none", choices=["none", "mixup", "cutmix"])
    ap.add_argument("--mix-alpha", type=float, default=1.0)
    ap.add_argument("--mix-prob", type=float, default=0.5)
    ap.add_argument("--ema", type=float, default=0.9995, help="EMA decay per step (0 = off)")
    ap.add_argument("--teacher", nargs="*", default=[], help="teacher checkpoints for distillation")
    ap.add_argument("--teacher-weights", nargs="*", type=float, default=None,
                    help="weights of the teachers' softened probabilities (default: equal)")
    ap.add_argument("--kd-alpha", type=float, default=0.9, help="weight of the KD term (1 = pure distillation)")
    ap.add_argument("--kd-temp", type=float, default=4.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "final.json")):  # chained Slurm jobs: training already finished
        print(f"{args.out} is already finished, nothing to do", flush=True)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    norm = Normalizer(dev)

    # ---- data
    x_tr, y_tr, x_va, y_va = load_split(args.data_dir, args.val_size)
    test = np.load(os.path.join(args.data_dir, "test.npz"))
    x_te = torch.from_numpy(test["x"]).permute(0, 3, 1, 2).contiguous()
    train_ds = ArrayDataset(x_tr, y_tr, make_train_transform(args.aug))
    loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0,
                        prefetch_factor=4 if args.workers > 0 else None)

    # ---- model
    overrides = {} if args.drop is None else {"drop": args.drop}
    if args.upsample:
        overrides["upsample"] = args.upsample
    model = build_model(args.arch, **overrides).to(dev).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"arch={args.arch} params={n_params:,} (limit {PARAM_LIMIT:,}) train={len(x_tr)} "
          f"val={0 if x_va is None else len(x_va)}", flush=True)
    if n_params > PARAM_LIMIT:
        print(f"WARNING: {args.arch} exceeds the {PARAM_LIMIT:,}-parameter limit (fine for a teacher only)", flush=True)
    teachers = [load_teacher(p, dev) for p in args.teacher]
    tw = torch.tensor(args.teacher_weights or [1.0] * len(teachers), device=dev)
    assert len(tw) == len(teachers), "--teacher-weights needs one weight per teacher"
    tw = (tw / tw.sum()).view(-1, 1, 1)
    if teachers:
        print(f"distilling from {len(teachers)} teacher(s): {args.teacher} weights {tw.flatten().tolist()}",
              flush=True)
    fwd = torch.compile(model) if args.compile else model

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.wd)
    steps_per_epoch = len(loader)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    ema = EMA(model, args.ema) if args.ema > 0 else None

    # ---- resume
    start_epoch, ckpt_path = 0, os.path.join(args.out, "ckpt_last.pt")
    log_path = os.path.join(args.out, "log.jsonl")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.model.load_state_dict(ck["ema"])
            ema.n = ck["ema_n"]
        start_epoch = ck["epoch"]
        with open(log_path) as f:  # drop log lines written after the checkpoint
            lines = f.readlines()[:start_epoch]
        with open(log_path, "w") as f:
            f.writelines(lines)
        print(f"resumed from epoch {start_epoch}", flush=True)
    else:
        with open(os.path.join(args.out, "args.json"), "w") as f:
            json.dump({**vars(args), "params": n_params}, f, indent=2)
        open(log_path, "w").close()

    # ---- train
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        tot_loss = torch.zeros((), device=dev)
        tot_correct = torch.zeros((), device=dev)
        n_seen = 0
        for xb, yb in loader:
            xb = norm(xb.to(dev, non_blocking=True))
            yb = yb.to(dev, non_blocking=True)
            if args.cutout > 0:
                xb = cutout(xb, args.cutout)
            lam, y2 = 1.0, yb
            if args.mix != "none" and np.random.rand() < args.mix_prob:
                xb, y2, lam = mix_batch(xb, yb, args.mix, args.mix_alpha)

            T, p_t = args.kd_temp, None
            if teachers:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    p_t = (torch.stack([t(xb).float().div(T).softmax(1) for t in teachers]) * tw).sum(0)

            def loss_fn():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = fwd(xb).float()
                ce = lam * F.cross_entropy(logits, yb, label_smoothing=args.label_smoothing) + \
                    (1 - lam) * F.cross_entropy(logits, y2, label_smoothing=args.label_smoothing)
                if p_t is None:
                    return logits, ce
                kd = F.kl_div(F.log_softmax(logits / T, 1), p_t, reduction="batchmean") * T * T
                return logits, args.kd_alpha * kd + (1 - args.kd_alpha) * ce

            logits, loss = loss_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.sam > 0:  # sharpness-aware minimization (Foret et al. 2021): gradient at w + rho * g / |g|
                with torch.no_grad():
                    ps = [p for p in model.parameters() if p.grad is not None]
                    gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad) for p in ps]))
                    eps = torch._foreach_mul([p.grad for p in ps], args.sam / (gnorm.item() + 1e-12))
                    torch._foreach_add_(ps, eps)
                opt.zero_grad(set_to_none=True)
                loss_fn()[1].backward()
                with torch.no_grad():
                    torch._foreach_sub_(ps, eps)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update()
            tot_loss += loss.detach() * len(yb)
            tot_correct += (logits.argmax(1) == yb).sum()
            n_seen += len(yb)

        rec = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_loss": tot_loss.item() / n_seen,
               "train_acc": tot_correct.item() / n_seen, "time": time.time() - t0}
        if x_va is not None:
            z = predict_logits(model, x_va, norm, tta=False)
            rec["val_loss"] = F.cross_entropy(z, y_va.to(dev)).item()
            rec["val_acc"] = accuracy(z, y_va)
            if ema is not None:
                z = predict_logits(ema.model, x_va, norm, tta=False)
                rec["val_loss_ema"] = F.cross_entropy(z, y_va.to(dev)).item()
                rec["val_acc_ema"] = accuracy(z, y_va)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)

        if (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "ema": ema.model.state_dict() if ema else None, "ema_n": ema.n if ema else 0,
                        "epoch": epoch + 1}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    # ---- final evaluation, test-set predictions, submission
    torch.save({"arch": args.arch, "arch_overrides": overrides, "params": n_params, "args": vars(args),
                "state_dict": model.state_dict(), "ema_state_dict": ema.model.state_dict() if ema else None},
               os.path.join(args.out, "model.pt"))
    final = {"params": n_params}
    candidates = {"model": model} if ema is None else {"model": model, "ema": ema.model}
    test_logits = {}
    for name, m in candidates.items():
        test_logits[name] = predict_logits(m, x_te, norm, tta=True).cpu().numpy()
        if x_va is not None:
            final[f"val_acc_{name}"] = accuracy(predict_logits(m, x_va, norm, tta=False), y_va)
            final[f"val_acc_{name}_tta"] = accuracy(predict_logits(m, x_va, norm, tta=True), y_va)
    np.savez(os.path.join(args.out, "test_logits.npz"), ids=test["ids"], **test_logits)
    best = "ema" if ema is not None else "model"
    pd.DataFrame({"id": test["ids"], "label": test_logits[best].argmax(1)}).to_csv(
        os.path.join(args.out, "submission.csv"), index=False)
    final["submission_from"] = best
    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("FINAL", json.dumps(final), flush=True)
    del loader  # shut down DataLoader worker processes


if __name__ == "__main__":
    main()
    # Persistent DataLoader workers can block interpreter shutdown (seen with Python 3.14), which would keep
    # the Slurm job (and its GPU) alive after training. All outputs are written at this point, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

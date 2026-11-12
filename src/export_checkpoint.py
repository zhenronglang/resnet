"""Export a slim, self-describing checkpoint (EMA weights only) from a training run.

    python src/export_checkpoint.py runs/<run>/model.pt checkpoints/final_model.pt
"""
import sys

import torch

from models import build_model, count_params

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["ema_state_dict"] if ck.get("ema_state_dict") else ck["state_dict"]
model = build_model(ck["arch"], **ck.get("arch_overrides", {}))
model.load_state_dict(sd)
torch.save({"arch": ck["arch"], "arch_overrides": ck.get("arch_overrides", {}), "params": count_params(model),
            "args": ck["args"], "state_dict": model.state_dict(), "source": src}, dst)
print(f"exported {src} -> {dst}: {ck['arch']}, {count_params(model):,} params")

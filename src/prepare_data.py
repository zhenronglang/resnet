"""Convert the competition PNGs into compact uint8 numpy arrays (one-time step).

Input  (data/):  train_labels.csv, train_images/*.png, test.csv, test_images/*.png
Output (data/):  train.npz  -> x: (50000, 32, 32, 3) uint8, y: (50000,) int64
                 test.npz   -> x: (2000, 32, 32, 3) uint8,  ids: (2000,) int64
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image


def load_images(folder, filenames):
    return np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")) for f in filenames])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    d = args.data_dir

    train = pd.read_csv(os.path.join(d, "train_labels.csv")).sort_values("id")
    x_train = load_images(os.path.join(d, "train_images"), train["filename"])
    y_train = train["label"].to_numpy(np.int64)
    np.savez(os.path.join(d, "train.npz"), x=x_train, y=y_train, ids=train["id"].to_numpy(np.int64))

    test = pd.read_csv(os.path.join(d, "test.csv")).sort_values("id")
    x_test = load_images(os.path.join(d, "test_images"), test["filename"])
    np.savez(os.path.join(d, "test.npz"), x=x_test, ids=test["id"].to_numpy(np.int64))

    print(f"train: {x_train.shape} {x_train.dtype}, labels {np.bincount(y_train)}")
    print(f"test:  {x_test.shape} {x_test.dtype}")
    for name, x in [("train", x_train), ("test", x_test)]:
        xf = x.reshape(-1, 3).astype(np.float64) / 255.0
        print(f"{name} per-channel mean {xf.mean(0).round(4)}, std {xf.std(0).round(4)}")


if __name__ == "__main__":
    main()

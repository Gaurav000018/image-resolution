"""Unzip DIV2K and convert training HR images to .npy for fast random cropping.

Expects data/raw/DIV2K_train_HR.zip and data/raw/DIV2K_valid_HR.zip
(download links: https://data.vision.ee.ethz.ch/cvl/DIV2K/).

Result:
    data/train_npy/*.npy        800 training images (uint8 HxWx3)
    data/DIV2K_valid_HR/*.png   100 validation/test images
"""
import argparse
import glob
import os
import zipfile

import numpy as np
from PIL import Image
from tqdm import tqdm


def unzip(zip_path, out_dir):
    if not os.path.exists(zip_path):
        print(f"[skip] {zip_path} not found")
        return
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print(f"[ok] extracted {zip_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--max-train", type=int, default=800, help="use fewer images to save disk/time")
    args = p.parse_args()

    raw = os.path.join(args.data, "raw")
    train_png = os.path.join(args.data, "DIV2K_train_HR")
    valid_png = os.path.join(args.data, "DIV2K_valid_HR")
    if not os.path.isdir(train_png):
        unzip(os.path.join(raw, "DIV2K_train_HR.zip"), args.data)
    if not os.path.isdir(valid_png):
        unzip(os.path.join(raw, "DIV2K_valid_HR.zip"), args.data)

    npy_dir = os.path.join(args.data, "train_npy")
    os.makedirs(npy_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(train_png, "*.png")))[: args.max_train]
    for f in tqdm(files, desc="png -> npy"):
        out = os.path.join(npy_dir, os.path.splitext(os.path.basename(f))[0] + ".npy")
        if not os.path.exists(out):
            np.save(out, np.asarray(Image.open(f).convert("RGB")))
    print(f"[ok] {len(files)} training images in {npy_dir}")
    print(f"[ok] {len(glob.glob(os.path.join(valid_png, '*.png')))} validation images in {valid_png}")


if __name__ == "__main__":
    main()

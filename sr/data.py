"""DIV2K dataset.

HR images are converted once to .npy files (prepare_data.py) and memory-mapped,
so random crops read only a few KB from disk instead of decoding a 2K PNG.
LR patches are generated on the fly with bicubic downscaling (the standard
"bicubic degradation" used in SR benchmarks).
"""
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degrade import degrade


class DIV2KTrain(Dataset):
    def __init__(self, npy_dir: str, scale: int = 4, lr_patch: int = 48,
                 repeat: int = 20, augment: bool = True, degradation: str = "bicubic"):
        self.files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {npy_dir}. Run prepare_data.py first.")
        self.scale, self.lr_patch, self.repeat, self.augment = scale, lr_patch, repeat, augment
        self.degradation = degradation
        self._cache = {}

    def __len__(self):
        return len(self.files) * self.repeat

    def _load(self, idx):
        # mmap handles are opened lazily inside each DataLoader worker
        if idx not in self._cache:
            self._cache[idx] = np.load(self.files[idx], mmap_mode="r")
        return self._cache[idx]

    def __getitem__(self, i):
        img = self._load(i % len(self.files))
        hp = self.lr_patch * self.scale
        h, w = img.shape[:2]
        y, x = random.randint(0, h - hp), random.randint(0, w - hp)
        hr = np.ascontiguousarray(img[y:y + hp, x:x + hp])

        if self.augment:
            if random.random() < 0.5:
                hr = hr[:, ::-1]
            if random.random() < 0.5:
                hr = hr[::-1, :]
            if random.random() < 0.5:
                hr = hr.transpose(1, 0, 2)
            hr = np.ascontiguousarray(hr)

        if self.degradation == "real":
            lr = np.asarray(degrade(Image.fromarray(hr), self.scale))
        else:
            lr = np.asarray(Image.fromarray(hr).resize((self.lr_patch, self.lr_patch), Image.BICUBIC))
        to_t = lambda a: torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
        return to_t(lr), to_t(hr)


def list_images(folder: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    return sorted(files)

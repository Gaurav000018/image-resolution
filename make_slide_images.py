"""Crops of real model outputs for the presentation -> slide_images/

Picks, over a handful of test images, the 256x256 window where the difference
between methods is largest, so the slides show a visible (but genuine) example.
"""
import io
import os

import numpy as np
import torch
from PIL import Image, ImageFilter

from sr.data import list_images
from sr.utils import (bicubic_downscale, bicubic_upscale, load_model, mod_crop,
                      pil_to_tensor, psnr, rgb_to_y, tensor_to_pil, upscale_tensor)

OUT = "slide_images"
os.makedirs(OUT, exist_ok=True)
S, WIN = 4, 256
M = {k: load_model(f"runs/{v}/best.pt")[0] for k, v in
     {"srcnn": "srcnn_x4", "edsr": "edsr_x4", "edsr_real": "edsr_x4_real"}.items()}
amp = dict(device_type="cuda", dtype=torch.bfloat16)
files = list_images("data/DIV2K_valid_HR")[10:40]


def run(model, lr):
    with torch.no_grad(), torch.autocast(**amp):
        return tensor_to_pil(upscale_tensor(M[model], pil_to_tensor(lr), S))


def mse_map(a, b):
    return (rgb_to_y(np.asarray(a)) - rgb_to_y(np.asarray(b))) ** 2


def best_window(gain):
    """top-left of the WIN x WIN window with the largest summed gain."""
    best, pos = -1e18, (0, 0)
    h, w = gain.shape
    ii = gain.cumsum(0).cumsum(1)
    for y in range(0, h - WIN, 32):
        for x in range(0, w - WIN, 32):
            s = ii[y + WIN - 1, x + WIN - 1] - (ii[y - 1, x + WIN - 1] if y else 0) \
                - (ii[y + WIN - 1, x - 1] if x else 0) + (ii[y - 1, x - 1] if x and y else 0)
            if s > best:
                best, pos = s, (x, y)
    return pos, best


def real_damage(hr):
    """A typical 'bad phone photo': mild blur, x4 down, sensor noise, JPEG."""
    img = hr.filter(ImageFilter.GaussianBlur(1.0))
    img = img.resize((hr.size[0] // S, hr.size[1] // S), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) + np.random.default_rng(0).normal(0, 10, (img.size[1], img.size[0], 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=55)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def save(img, box, name):
    img.crop(box).resize((512, 512), Image.LANCZOS).save(f"{OUT}/{name}.png")


# ---------- clean: largest EDSR gain over bicubic ----------
cands = []
for f in files:
    hr = mod_crop(Image.open(f).convert("RGB"), S)
    lr = bicubic_downscale(hr, S)
    bic, ed = bicubic_upscale(lr, S), run("edsr", lr)
    (x, y), g = best_window(mse_map(bic, hr) - mse_map(ed, hr))
    cands.append((g, f, (x, y)))
g, f, (x, y) = max(cands)
hr = mod_crop(Image.open(f).convert("RGB"), S)
lr = bicubic_downscale(hr, S)
box = (x, y, x + WIN, y + WIN)
for k, v in {"hr": hr, "bicubic": bicubic_upscale(lr, S), "srcnn": run("srcnn", lr), "edsr": run("edsr", lr)}.items():
    save(v, box, f"clean_{k}")
lr.crop(tuple(c // S for c in box)).resize((512, 512), Image.NEAREST).save(f"{OUT}/clean_lr_pixels.png")
print("clean example:", os.path.basename(f), box)

# ---------- real-world: largest EDSR-real gain over EDSR ----------
cands = []
for f in files:
    hr = mod_crop(Image.open(f).convert("RGB"), S)
    lr = real_damage(hr)
    ed, er = run("edsr", lr), run("edsr_real", lr)
    (x, y), g = best_window(mse_map(ed, hr) - mse_map(er, hr))
    cands.append((g, f, (x, y)))
g, f, (x, y) = max(cands)
hr = mod_crop(Image.open(f).convert("RGB"), S)
lr = real_damage(hr)
box = (x, y, x + WIN, y + WIN)
outs = {"hr": hr, "input": bicubic_upscale(lr, S), "edsr": run("edsr", lr), "edsr_real": run("edsr_real", lr)}
for k, v in outs.items():
    save(v, box, f"real_{k}")
print("real example:", os.path.basename(f), box,
      {k: round(psnr(np.asarray(v), np.asarray(hr), S), 2) for k, v in outs.items() if k != "hr"})

# ---------- title slide: one scene, left half blurry (bicubic), right half EDSR ----------
hr3 = mod_crop(Image.open("data/DIV2K_valid_HR/0811.png").convert("RGB"), S)
lr3 = bicubic_downscale(hr3, S)
b3 = (400, 250, 1360, 970)
blur, sharp = bicubic_upscale(lr3, S).crop(b3), run("edsr", lr3).crop(b3)
w, h = blur.size
hero = blur.copy()
hero.paste(sharp.crop((w // 2, 0, w, h)), (w // 2, 0))
hero.save(f"{OUT}/hero.png")
print("done")

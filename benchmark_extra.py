"""Extra measurements for the report:
  * per-image inference time (510x339 -> 2040x1356) for bicubic and each model
  * EDSR-real on the clean bicubic test set (90 images)
Writes results/extra.json
"""
import json
import time

import numpy as np
import torch
from PIL import Image

from sr.data import list_images
from sr.utils import (bicubic_downscale, bicubic_upscale, load_model, mod_crop,
                      pil_to_tensor, psnr, ssim, tensor_to_uint8, upscale_tensor)

SCALE = 4
files = list_images("data/DIV2K_valid_HR")[10:]
ckpts = {"SRCNN": "runs/srcnn_x4/best.pt", "EDSR": "runs/edsr_x4/best.pt",
         "EDSR-real": "runs/edsr_x4_real/best.pt"}
models = {k: load_model(v)[0] for k, v in ckpts.items()}
amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available())
out = {"timing_ms": {}, "edsr_real_clean": {}}

# timing on 10 images, after 2 warm-up runs
lrs = [pil_to_tensor(bicubic_downscale(mod_crop(Image.open(f).convert("RGB"), SCALE), SCALE)) for f in files[:10]]
lr_pils = [bicubic_downscale(mod_crop(Image.open(f).convert("RGB"), SCALE), SCALE) for f in files[:10]]
t = time.perf_counter()
for im in lr_pils:
    bicubic_upscale(im, SCALE)
out["timing_ms"]["Bicubic"] = (time.perf_counter() - t) / len(lr_pils) * 1000
for name, m in models.items():
    with torch.no_grad(), torch.autocast(**amp):
        for x in lrs[:2]:
            upscale_tensor(m, x, SCALE)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for x in lrs:
            upscale_tensor(m, x, SCALE)
        torch.cuda.synchronize()
    out["timing_ms"][name] = (time.perf_counter() - t) / len(lrs) * 1000
print("timing", out["timing_ms"], flush=True)

# EDSR-real on clean inputs
ps, ss = [], []
for f in files:
    hr = mod_crop(Image.open(f).convert("RGB"), SCALE)
    lr = bicubic_downscale(hr, SCALE)
    with torch.no_grad(), torch.autocast(**amp):
        sr = tensor_to_uint8(upscale_tensor(models["EDSR-real"], pil_to_tensor(lr), SCALE))
    h = np.asarray(hr)
    ps.append(psnr(sr, h, SCALE))
    ss.append(ssim(sr, h, SCALE))
out["edsr_real_clean"] = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)), "n": len(files)}
print("edsr_real_clean", out["edsr_real_clean"], flush=True)
json.dump(out, open("results/extra.json", "w"), indent=2)

"""Upscale any image with a trained model.

    python upscale.py input.jpg --checkpoint runs/edsr_x4/best.pt --out output.png
"""
import argparse
import os

import torch
from PIL import Image

from sr.utils import load_model, pil_to_tensor, tensor_to_pil, upscale_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--checkpoint", default="runs/edsr_x4/best.pt")
    p.add_argument("--out", default="")
    args = p.parse_args()

    model, scale, name = load_model(args.checkpoint)
    img = Image.open(args.input).convert("RGB")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        sr = upscale_tensor(model, pil_to_tensor(img), scale)
    out = args.out or os.path.splitext(args.input)[0] + f"_{name}_x{scale}.png"
    tensor_to_pil(sr).save(out)
    print(f"{img.size} -> {sr.shape[-1]}x{sr.shape[-2]} saved to {out}")


if __name__ == "__main__":
    main()

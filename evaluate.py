"""Evaluate bicubic vs trained models on held-out DIV2K validation images.

The first 10 validation images are used for checkpoint selection during
training, so by default the test set is the remaining 90 images.

    python evaluate.py --checkpoints runs/srcnn_x4/best.pt runs/edsr_x4/best.pt

Writes results/metrics.md, results/metrics.csv, side-by-side comparison
figures in results/comparisons/ and training curves in results/curves.png.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sr.data import list_images
from sr.models import count_params
from sr.utils import (bicubic_upscale, load_model, make_lr, mod_crop, pil_to_tensor,
                      psnr, ssim, tensor_to_uint8, upscale_tensor)


def comparison_figure(hr, lr, outputs, path, scale, crop=192):
    """Full HR image + zoomed crop of each method."""
    h, w = hr.shape[:2]
    # crop around the most textured region (highest local variance) for an informative zoom
    gray = hr.mean(axis=2)
    best, by, bx = -1, 0, 0
    for y in range(0, h - crop, crop // 2):
        for x in range(0, w - crop, crop // 2):
            v = gray[y:y + crop, x:x + crop].var()
            if v > best:
                best, by, bx = v, y, x
    panels = [("HR (ground truth)", hr)] + list(outputs.items())
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.2 * (len(panels) + 1), 3.6))
    axes[0].imshow(hr)
    axes[0].add_patch(plt.Rectangle((bx, by), crop, crop, fill=False, ec="red", lw=2))
    axes[0].set_title("Full image")
    for ax, (name, img) in zip(axes[1:], panels):
        ax.imshow(img[by:by + crop, bx:bx + crop], interpolation="nearest")
        if name.startswith("HR"):
            ax.set_title(name)
        else:
            ax.set_title(f"{name}\n{psnr(img, hr, scale):.2f} dB (full image)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_curves(run_dirs, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    any_plotted = False
    for d in run_dirs:
        log = os.path.join(d, "log.csv")
        if not os.path.exists(log):
            continue
        rows = list(csv.DictReader(open(log)))
        if not rows:
            continue
        it = [int(r["iter"]) for r in rows]
        name = os.path.basename(d)
        a1.plot(it, [float(r["loss"]) for r in rows], label=name)
        a2.plot(it, [float(r["val_psnr"]) for r in rows], label=name)
        any_plotted = True
    if not any_plotted:
        plt.close(fig)
        return
    a1.set(title="Training L1 loss", xlabel="iteration")
    a2.set(title="Validation PSNR (dB)", xlabel="iteration")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", default=[])
    p.add_argument("--data", default="data/DIV2K_valid_HR")
    p.add_argument("--skip", type=int, default=10, help="skip images used for validation")
    p.add_argument("--limit", type=int, default=0, help="evaluate only N images (0 = all)")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--figures", type=int, default=5)
    p.add_argument("--degradation", default="bicubic", choices=["bicubic", "real"],
                   help="test inputs: clean bicubic, or real-world blur+noise+JPEG (fixed seed)")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "comparisons"), exist_ok=True)
    files = list_images(args.data)[args.skip:]
    if args.limit:
        files = files[: args.limit]

    models = {}
    for ck in args.checkpoints:
        m, s, name = load_model(ck)
        assert s == args.scale, f"{ck} is x{s}, but --scale is {args.scale}"
        models[name.upper()] = m

    methods = ["Bicubic"] + list(models)
    scores = {m: {"psnr": [], "ssim": []} for m in methods}
    use_amp = torch.cuda.is_available()

    for i, f in enumerate(tqdm(files, desc="evaluating")):
        hr_img = mod_crop(Image.open(f).convert("RGB"), args.scale)
        lr_img = make_lr(hr_img, args.scale, args.degradation, seed=1000 + i)
        hr = np.asarray(hr_img)
        outputs = {"Bicubic": np.asarray(bicubic_upscale(lr_img, args.scale))}
        for name, m in models.items():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs[name] = tensor_to_uint8(upscale_tensor(m, pil_to_tensor(lr_img), args.scale))
        for name, out in outputs.items():
            scores[name]["psnr"].append(psnr(out, hr, args.scale))
            scores[name]["ssim"].append(ssim(out, hr, args.scale))
        if i < args.figures:
            base = os.path.splitext(os.path.basename(f))[0]
            comparison_figure(hr, np.asarray(lr_img), outputs,
                              os.path.join(args.out, "comparisons", f"{base}.png"), args.scale)

    params = {"Bicubic": "-"}
    params.update({n: f"{count_params(m) / 1e6:.2f}M" for n, m in models.items()})
    rows = [(m, np.mean(scores[m]["psnr"]), np.mean(scores[m]["ssim"]), params[m]) for m in methods]

    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "psnr_db", "ssim", "params"])
        for r in rows:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.4f}", r[3]])
    table = [f"Test set: {len(files)} DIV2K validation images, x{args.scale}, "
             f"{args.degradation} degradation, "
             f"PSNR/SSIM on Y channel, {args.scale}px border shaved\n",
             "| Method | PSNR (dB) | SSIM | Params |", "|---|---|---|---|"]
    table += [f"| {r[0]} | {r[1]:.2f} | {r[2]:.4f} | {r[3]} |" for r in rows]
    md = "\n".join(table)
    open(os.path.join(args.out, "metrics.md"), "w").write(md + "\n")
    print("\n" + md)

    run_dirs = sorted({os.path.dirname(os.path.abspath(c)) for c in args.checkpoints})
    plot_curves(run_dirs, os.path.join(args.out, "curves.png"))
    print(f"\nfigures -> {args.out}/comparisons/, curves -> {args.out}/curves.png")


if __name__ == "__main__":
    main()

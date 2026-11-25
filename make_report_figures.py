"""Figures for the IEEE report (same grayscale/serif style as the original template).
    report_figs/fig_arch.png  (831x780)   Fig. 1 system architecture
    report_figs/fig_psnr.png  (938x584)   Fig. 2 PSNR-Y per model, clean vs real-world
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Serif", "hatch.linewidth": 1.6})
os.makedirs("report_figs", exist_ok=True)
DPI = 100


def box(ax, x, y, w, h, text, dark=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc="#d9d9d9" if dark else "#efefef", ec="black", lw=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=20, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=2, mutation_scale=28, color="black"))


def fig_arch():
    fig = plt.figure(figsize=(8.31, 7.80), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.31)
    ax.set_ylim(0, 7.80)
    ax.axis("off")
    box(ax, 0.3, 6.35, 3.45, 1.15, "DIV2K HR images\n(400 train, 100 val)")
    box(ax, 4.55, 6.35, 3.45, 1.15, "LR degradation\nbicubic or real-world")
    box(ax, 0.3, 4.4, 7.7, 1.3, "Training engine (L1 loss, Adam)\nSRCNN  |  EDSR  |  EDSR-real  (x4)", dark=True)
    box(ax, 0.3, 2.6, 3.45, 1.15, "Checkpoints\n(best.pt / last.pt)")
    box(ax, 4.55, 2.6, 3.45, 1.15, "Evaluation\nPSNR-Y / SSIM-Y")
    box(ax, 0.3, 0.8, 3.45, 1.15, "Inference module\n(tiled, shared)", dark=True)
    box(ax, 4.55, 0.8, 3.45, 1.15, "Gradio web demo\n+ CLI (upscale.py)")
    arrow(ax, 3.75, 6.92, 4.55, 6.92)
    arrow(ax, 2.02, 6.35, 2.02, 5.7)
    arrow(ax, 6.27, 6.35, 6.27, 5.7)
    arrow(ax, 2.02, 4.4, 2.02, 3.75)
    arrow(ax, 3.75, 3.17, 4.55, 3.17)
    arrow(ax, 2.02, 2.6, 2.02, 1.95)
    arrow(ax, 3.75, 1.37, 4.55, 1.37)
    fig.savefig("report_figs/fig_arch.png", dpi=DPI)
    plt.close(fig)


def read_metrics(path):
    return {r["method"]: float(r["psnr_db"]) for r in csv.DictReader(open(path))}


def fig_psnr():
    clean = read_metrics("results/metrics.csv")
    real = read_metrics("results_real/metrics.csv")
    extra = json.load(open("results/extra.json"))
    clean["EDSR-REAL"] = extra["edsr_real_clean"]["psnr"]
    names = ["Bicubic", "SRCNN", "EDSR", "EDSR-REAL"]
    labels = ["Bicubic", "SRCNN", "EDSR", "EDSR-real"]

    fig, ax = plt.subplots(figsize=(9.38, 5.84), dpi=DPI)
    w = 0.36
    xs = range(len(names))
    b1 = ax.bar([x - w / 2 for x in xs], [clean[n] for n in names], w, color="#bdbdbd",
                ec="black", lw=1.5, label="Clean bicubic ×4")
    b2 = ax.bar([x + w / 2 for x in xs], [real[n] for n in names], w, color="white",
                ec="black", lw=1.5, hatch="//", label="Real-world ×4")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.1f}",
                    ha="center", va="bottom", fontsize=14)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=17)
    ax.set_ylabel("PSNR-Y (dB)", fontsize=17)
    ax.set_ylim(24, 31)
    ax.tick_params(axis="y", labelsize=14, width=1.5)
    ax.tick_params(axis="x", width=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.5)
    ax.legend(frameon=False, fontsize=15, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig("report_figs/fig_psnr.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig_arch()
    if os.path.exists("results/extra.json"):
        fig_psnr()
    print("figures written to report_figs/")

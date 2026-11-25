"""Train a super-resolution model on DIV2K.

Examples:
    python train.py --model edsr  --scale 4 --iters 30000
    python train.py --model srcnn --scale 4 --iters 10000
    python train.py --model edsr  --scale 4 --minutes 60     # stop after 1 hour
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from sr.data import DIV2KTrain, list_images
from sr.models import build_model, count_params
from sr.utils import (make_lr, mod_crop, pil_to_tensor, psnr, tensor_to_uint8,
                      upscale_tensor)


@torch.no_grad()
def validate(model, files, scale, degradation="bicubic"):
    model.eval()
    scores = []
    for i, f in enumerate(files):
        hr = mod_crop(Image.open(f).convert("RGB"), scale)
        lr = make_lr(hr, scale, degradation, seed=i)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            sr = upscale_tensor(model, pil_to_tensor(lr), scale)
        scores.append(psnr(tensor_to_uint8(sr), np.asarray(hr), shave=scale))
    model.train()
    return float(np.mean(scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="edsr", choices=["srcnn", "edsr", "edsr-large"])
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--data", default="data")
    p.add_argument("--iters", type=int, default=30000)
    p.add_argument("--minutes", type=float, default=0, help="optional wall-clock limit (0 = none)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patch", type=int, default=48, help="LR patch size")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-images", type=int, default=10)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default="", help="continue an interrupted run (weights + optimizer)")
    p.add_argument("--init", default="", help="start from these weights (fine-tuning)")
    p.add_argument("--degradation", default="bicubic", choices=["bicubic", "real"],
                   help="'real' = random blur + noise + JPEG (real-world photos)")
    args = p.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    suffix = "_real" if args.degradation == "real" else ""
    run_dir = os.path.join(args.out, f"{args.model}_x{args.scale}{suffix}")
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(args.model, args.scale).to(device)
    print(f"{args.model} x{args.scale} ({args.degradation}): "
          f"{count_params(model) / 1e6:.2f}M params on {device}")
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["state_dict"])
        print(f"initialised from {args.init}")

    ds = DIV2KTrain(os.path.join(args.data, "train_npy"), args.scale, args.patch,
                    degradation=args.degradation)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_files = list_images(os.path.join(args.data, "DIV2K_valid_HR"))[: args.val_images]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-7)
    loss_fn = nn.L1Loss()

    start_it, best = 0, 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_it, best = ck["iter"], ck.get("best_psnr", 0.0)
        print(f"resumed from iter {start_it}")

    def save(path, it, best_psnr):
        torch.save({"model_name": args.model, "scale": args.scale, "iter": it,
                    "degradation": args.degradation,
                    "best_psnr": best_psnr, "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(), "scheduler": sched.state_dict()}, path)

    log_path = os.path.join(run_dir, "log.csv")
    new_log = not os.path.exists(log_path) or not args.resume
    log_f = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["iter", "loss", "val_psnr", "lr", "minutes"])

    model.train()
    it, t0, running = start_it, time.time(), []
    done = False
    while not done:
        for lr_img, hr_img in dl:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                sr = model(lr_img)
            loss = loss_fn(sr.float(), hr_img)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running.append(loss.item())
            it += 1

            minutes = (time.time() - t0) / 60
            out_of_time = args.minutes > 0 and minutes >= args.minutes
            if it % args.val_every == 0 or it >= args.iters or out_of_time:
                v = validate(model, val_files, args.scale, args.degradation)
                avg = float(np.mean(running))
                running = []
                log.writerow([it, f"{avg:.5f}", f"{v:.3f}", f"{sched.get_last_lr()[0]:.2e}", f"{minutes:.1f}"])
                log_f.flush()
                tag = ""
                if v > best:
                    best = v
                    save(os.path.join(run_dir, "best.pt"), it, best)
                    tag = "  * new best"
                save(os.path.join(run_dir, "last.pt"), it, best)
                print(f"iter {it:6d} | loss {avg:.4f} | val PSNR {v:.2f} dB | {minutes:5.1f} min{tag}", flush=True)
            if it >= args.iters or out_of_time:
                done = True
                break

    log_f.close()
    print(f"done. best val PSNR {best:.2f} dB -> {os.path.join(run_dir, 'best.pt')}")


if __name__ == "__main__":
    main()

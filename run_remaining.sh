#!/usr/bin/env bash
# Runs after the bicubic EDSR training: fine-tune EDSR-real, then evaluate everything.
cd "$(dirname "$0")"
PY=.venv/Scripts/python

until grep -q "^done\. best val PSNR .*edsr_x4" train.log; do
  grep -q "Traceback" train.log && { echo "EDSR training failed"; exit 1; }
  sleep 20
done
echo "EDSR bicubic training finished"

# fine-tune on real-world degradations (GPU) while evaluating bicubic models in parallel
$PY -u train.py --model edsr --degradation real --init runs/edsr_x4/best.pt \
    --iters 12000 --lr 1e-4 --val-every 1000 > train_real.log 2>&1 &
TRAIN_PID=$!

$PY evaluate.py --checkpoints runs/srcnn_x4/best.pt runs/edsr_x4/best.pt --out results \
    > eval_bicubic.log 2>&1 && echo "bicubic evaluation done" || echo "bicubic evaluation FAILED"

wait $TRAIN_PID && echo "EDSR-real training finished" || { echo "EDSR-real training FAILED"; exit 1; }

$PY evaluate.py --degradation real \
    --checkpoints runs/srcnn_x4/best.pt runs/edsr_x4/best.pt runs/edsr_x4_real/best.pt \
    --out results_real > eval_real.log 2>&1 && echo "real-world evaluation done" || echo "real-world evaluation FAILED"
echo ALL_DONE

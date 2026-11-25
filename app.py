"""Web demo: upload an image and compare bicubic vs the trained models.

    python app.py        -> open http://127.0.0.1:7860

Three modes:
  * "Upscale my image"             - the uploaded image is treated as the low-res input.
  * "Simulate clean low-res"       - the image is bicubic-downscaled, then restored,
                                     so you can compare against the original (with PSNR).
  * "Simulate real-world low-res"  - same, but with random blur + noise + JPEG added.

Use an EDSR-REAL model for real photos (phone pictures, compressed web images).
"""
import glob
import os

import gradio as gr
import numpy as np
import torch
from PIL import Image

from sr.utils import (bicubic_upscale, load_model, make_lr, mod_crop, pil_to_tensor,
                      psnr, tensor_to_pil, upscale_tensor)

MODES = ["Upscale my image", "Simulate clean low-res", "Simulate real-world low-res (blur+noise+JPEG)"]

MAX_INPUT_SIDE = 800  # keep the demo responsive

MODELS = {}
for ck in sorted(glob.glob("runs/*/best.pt")):
    m, s, name = load_model(ck)
    MODELS[f"{name.upper()} x{s}"] = (m, s)
if not MODELS:
    raise SystemExit("No trained models found in runs/*/best.pt. Train one with train.py first.")


def run(image, model_key, mode):
    if image is None:
        return None, None, ""
    model, scale = MODELS[model_key]
    img = image.convert("RGB")
    hr = None
    if mode != MODES[0]:
        hr = mod_crop(img, scale)
        img = make_lr(hr, scale, "real" if mode == MODES[2] else "bicubic",
                      seed=np.random.randint(1 << 30))
    if max(img.size) > MAX_INPUT_SIDE:
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.BICUBIC)
        hr = None

    bic = bicubic_upscale(img, scale)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        sr = tensor_to_pil(upscale_tensor(model, pil_to_tensor(img), scale))

    info = f"Input {img.size[0]}x{img.size[1]}  ->  output {sr.size[0]}x{sr.size[1]}"
    if hr is not None:
        h = np.asarray(hr)
        info += (f"\nPSNR vs original:  bicubic {psnr(np.asarray(bic), h, scale):.2f} dB  |  "
                 f"{model_key} {psnr(np.asarray(sr), h, scale):.2f} dB")
    return bic, sr, info


with gr.Blocks(title="Image Super-Resolution") as demo:
    gr.Markdown("# Image Super-Resolution\nReconstruct a high-resolution image from a low-resolution input.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Input image")
            default = next((k for k in MODELS if "REAL" in k), next((k for k in MODELS if k.startswith("EDSR")), list(MODELS)[0]))
            model_dd = gr.Dropdown(list(MODELS), value=default, label="Model")
            mode = gr.Radio(MODES, value=MODES[1], label="Mode")
            btn = gr.Button("Run", variant="primary")
            info = gr.Textbox(label="Info", lines=2)
        with gr.Column():
            out_bic = gr.Image(type="pil", label="Bicubic", format="png")
            out_sr = gr.Image(type="pil", label="Super-resolved", format="png")
    examples = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[10:14]
    if examples:
        gr.Examples(examples, inputs=inp)
    btn.click(run, [inp, model_dd, mode], [out_bic, out_sr, info])

if __name__ == "__main__":
    demo.launch()

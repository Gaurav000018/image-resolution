"""Image helpers, degradation, metrics and tiled inference."""
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


# ---------- conversion ----------

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float tensor (1, 3, H, W) in [0, 1]."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((arr * 255.0).round().astype(np.uint8))


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


# ---------- degradation ----------

def mod_crop(img: Image.Image, scale: int) -> Image.Image:
    """Crop so width/height are divisible by `scale`."""
    w, h = img.size
    return img.crop((0, 0, w - w % scale, h - h % scale))


def bicubic_downscale(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)


def bicubic_upscale(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.BICUBIC)


# ---------- metrics (standard SR protocol: Y channel, border shaved by `scale`) ----------

def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> Y channel (ITU-R BT.601, range 16-235) as float64."""
    img = img.astype(np.float64)
    return 16.0 + (65.481 * img[..., 0] + 128.553 * img[..., 1] + 24.966 * img[..., 2]) / 255.0


def _prep(sr: np.ndarray, hr: np.ndarray, shave: int):
    sr_y, hr_y = rgb_to_y(sr), rgb_to_y(hr)
    if shave > 0:
        sr_y = sr_y[shave:-shave, shave:-shave]
        hr_y = hr_y[shave:-shave, shave:-shave]
    return sr_y, hr_y


def psnr(sr: np.ndarray, hr: np.ndarray, shave: int = 0) -> float:
    sr_y, hr_y = _prep(sr, hr, shave)
    mse = np.mean((sr_y - hr_y) ** 2)
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(sr: np.ndarray, hr: np.ndarray, shave: int = 0) -> float:
    sr_y, hr_y = _prep(sr, hr, shave)
    return float(structural_similarity(sr_y, hr_y, data_range=255.0, gaussian_weights=True,
                                       sigma=1.5, use_sample_covariance=False))


# ---------- inference ----------

@torch.no_grad()
def upscale_tensor(model, lr: torch.Tensor, scale: int, tile: int = 256, overlap: int = 16) -> torch.Tensor:
    """Run `model` on `lr` (1, 3, h, w), splitting into overlapping tiles so big
    images fit in GPU memory. Returns (1, 3, h*scale, w*scale) on CPU."""
    device = next(model.parameters()).device
    _, c, h, w = lr.shape
    if h <= tile and w <= tile:
        return model(lr.to(device)).float().cpu()

    out = torch.zeros(1, c, h * scale, w * scale)
    weight = torch.zeros_like(out)
    step = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            y0, x0 = max(y, 0), max(x, 0)
            patch = lr[:, :, y0:y0 + tile, x0:x0 + tile].to(device)
            sr = model(patch).float().cpu()
            ph, pw = sr.shape[-2:]
            out[:, :, y0 * scale:y0 * scale + ph, x0 * scale:x0 * scale + pw] += sr
            weight[:, :, y0 * scale:y0 * scale + ph, x0 * scale:x0 * scale + pw] += 1
    return out / weight


def load_model(checkpoint_path: str, device: str = None):
    """Load a checkpoint written by train.py. Returns (model, scale, label),
    where label is e.g. 'edsr' or 'edsr-real' (trained on real-world degradations)."""
    from .models import build_model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], ckpt["scale"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    label = ckpt["model_name"]
    if ckpt.get("degradation", "bicubic") == "real":
        label += "-real"
    return model, ckpt["scale"], label


def make_lr(hr: Image.Image, scale: int, degradation: str = "bicubic", seed: int = 0) -> Image.Image:
    """LR test input. 'real' uses a fixed seed so every method sees the same degradation."""
    if degradation == "real":
        import random
        from .degrade import degrade
        return degrade(hr, scale, random.Random(seed))
    return bicubic_downscale(hr, scale)

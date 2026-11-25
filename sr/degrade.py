"""Real-world degradation pipeline (simplified, Real-ESRGAN style first-order model).

    HR --blur--> --downscale (random kernel)--> --noise--> --JPEG--> LR

Training on these random degradations teaches the network to remove blur,
sensor noise and compression artefacts while it upscales, so it works on
real photos and not only on clean bicubic-downscaled images.
"""
import io
import random

import numpy as np
from PIL import Image, ImageFilter

RESAMPLE = [Image.BICUBIC, Image.BILINEAR, Image.BOX, Image.LANCZOS]


def _gaussian_kernel(size, sx, sy, theta):
    """Anisotropic Gaussian kernel (size x size), rotated by theta."""
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(theta), np.sin(theta)
    xr, yr = c * xx + s * yy, -s * xx + c * yy
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return k / k.sum()


def random_blur(img: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:  # isotropic
        return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 2.5)))
    size = 21
    k = _gaussian_kernel(size, rng.uniform(0.3, 3.0), rng.uniform(0.3, 3.0), rng.uniform(0, np.pi))
    arr = np.asarray(img, dtype=np.float32)
    pad = size // 2
    p = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    # separable trick does not apply to rotated kernels; use FFT convolution per channel
    out = np.empty_like(arr)
    H, W = arr.shape[:2]
    kf = np.fft.rfft2(k, s=p.shape[:2])
    for ch in range(3):
        conv = np.fft.irfft2(np.fft.rfft2(p[..., ch]) * kf, s=p.shape[:2])
        out[..., ch] = conv[2 * pad:2 * pad + H, 2 * pad:2 * pad + W]
    return Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8))


def random_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    nrng = np.random.default_rng(rng.randrange(2 ** 32))
    sigma = rng.uniform(1, 20)
    if rng.random() < 0.4:  # gray (luminance) noise
        noise = nrng.normal(0, sigma, arr.shape[:2])[..., None]
    else:  # colour noise
        noise = nrng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).round().astype(np.uint8))


def random_jpeg(img: Image.Image, rng: random.Random) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(30, 95))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade(hr: Image.Image, scale: int, rng: random.Random = None) -> Image.Image:
    """Apply a random real-world degradation and downscale by `scale`."""
    rng = rng or random
    img = hr
    if rng.random() < 0.8:
        img = random_blur(img, rng)
    w, h = hr.size
    img = img.resize((w // scale, h // scale), rng.choice(RESAMPLE))
    if rng.random() < 0.7:
        img = random_noise(img, rng)
    if rng.random() < 0.7:
        img = random_jpeg(img, rng)
    return img

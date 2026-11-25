"""Super-resolution models.

All models share one interface: input is a low-resolution RGB tensor in [0, 1]
of shape (N, 3, h, w), output is the high-resolution tensor (N, 3, h*scale, w*scale).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Mean RGB of the DIV2K training set (used by EDSR for input normalisation).
DIV2K_MEAN = (0.4488, 0.4371, 0.4040)


class SRCNN(nn.Module):
    """SRCNN (Dong et al., 2014): 9-5-5 three-layer CNN.

    The original works on a bicubic-upscaled input; we do the bicubic upscaling
    inside forward() so it matches the LR -> HR interface of the other models.
    """

    def __init__(self, scale: int = 4, channels: int = 3):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, 64, 9, padding=4), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, channels, 5, padding=2),
        )

    def forward(self, x):
        up = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return up + self.body(up)  # residual learning: predict the missing detail


class ResBlock(nn.Module):
    def __init__(self, n_feats: int, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """Sub-pixel convolution (PixelShuffle) upsampler for scale 2, 3, 4 or 8."""

    def __init__(self, scale: int, n_feats: int):
        layers = []
        if scale & (scale - 1) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                layers += [nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1), nn.PixelShuffle(2)]
        elif scale == 3:
            layers += [nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f"Unsupported scale {scale}")
        super().__init__(*layers)


class EDSR(nn.Module):
    """EDSR (Lim et al., 2017). Defaults = the 'EDSR-baseline' configuration."""

    def __init__(self, scale: int = 4, n_resblocks: int = 16, n_feats: int = 64,
                 res_scale: float = 1.0, channels: int = 3):
        super().__init__()
        self.scale = scale
        self.register_buffer("mean", torch.tensor(DIV2K_MEAN).view(1, 3, 1, 1))
        self.head = nn.Conv2d(channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[ResBlock(n_feats, res_scale) for _ in range(n_resblocks)],
            nn.Conv2d(n_feats, n_feats, 3, padding=1),
        )
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, channels, 3, padding=1),
        )

    def forward(self, x):
        x = x - self.mean
        x = self.head(x)
        x = x + self.body(x)  # global residual
        x = self.tail(x)
        return x + self.mean


def build_model(name: str, scale: int) -> nn.Module:
    name = name.lower()
    if name == "srcnn":
        return SRCNN(scale)
    if name == "edsr":
        return EDSR(scale)
    if name == "edsr-large":
        return EDSR(scale, n_resblocks=32, n_feats=256, res_scale=0.1)
    raise ValueError(f"Unknown model '{name}'. Choose srcnn, edsr or edsr-large.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

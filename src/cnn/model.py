"""
model.py - U-Net that maps a noisy log-magnitude spectrogram to a clean one.

Input is one channel, (batch, 1, 257, 376). Frequency is padded 257 -> 264 and
time 376 -> 384 so three pooling stages land on integers, then the output is
cropped back. target="mask" applies a sigmoid; target="map" leaves the output
unbounded so it can be trained as log1p(clean magnitude).
"""
import torch
from torch import nn


FREQ, TIME = 257, 376
PAD_FREQ, PAD_TIME = 264, 384


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class DenoiseUNet(nn.Module):
    def __init__(self, base_channels=16, dropout=0.0, target="mask"):
        super().__init__()
        if target not in ("mask", "map"):
            raise ValueError(f"target must be 'mask' or 'map', got {target!r}")
        self.target = target
        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 4, c * 4)
        self.drop = nn.Dropout2d(dropout)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c * 4 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x):
        if x.shape[-2:] != (FREQ, TIME):
            raise ValueError(f"expected spectrogram {(FREQ, TIME)}, got {tuple(x.shape[-2:])}")
        # Pad on the right (time) and bottom (frequency). Crop those rows off at the end.
        x = nn.functional.pad(x, (0, PAD_TIME - TIME, 0, PAD_FREQ - FREQ))
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        h = self.drop(self.bottleneck(self.pool(s3)))
        h = self.dec3(torch.cat([self.up(h)[..., : s3.shape[-2], : s3.shape[-1]], s3], dim=1))
        h = self.dec2(torch.cat([self.up(h)[..., : s2.shape[-2], : s2.shape[-1]], s2], dim=1))
        h = self.dec1(torch.cat([self.up(h)[..., : s1.shape[-2], : s1.shape[-1]], s1], dim=1))
        y = self.out(h)[..., :FREQ, :TIME]
        if self.target == "mask":
            y = torch.sigmoid(y)
        return y

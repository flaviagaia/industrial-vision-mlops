"""U-Net compacta treinada do zero; quatro canais sigmoid independentes."""

import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        b = base_channels
        if b < 4 or b % 4:
            raise ValueError("base_channels deve ser múltiplo de 4")
        self.encoders = nn.ModuleList([block(3, b), block(b, b * 2), block(b * 2, b * 4)])
        self.bottleneck = block(b * 4, b * 8)
        self.decoders = nn.ModuleList([block(b * 12, b * 4), block(b * 6, b * 2), block(b * 3, b)])
        self.head = nn.Conv2d(b, 4, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


def segmentation_loss(logits, target):
    """BCE + soft Dice; sigmoid não deve ser aplicada antes da BCEWithLogits."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    dims = (0, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return bce + 1 - dice.mean()

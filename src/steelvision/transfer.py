"""Encoder ResNet18 opcional com normalização ImageNet incorporada ao modelo."""

import torch
from torch import nn
from torch.nn import functional as F

from .model import UNet, block


class ResNetUNet(nn.Module):
    def __init__(self, pretrained=False, train_encoder=False):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool = encoder.maxpool
        self.layers = nn.ModuleList(
            [encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4]
        )
        # Extração de características congelada é o padrão do experimento curto.
        # Fine-tuning deve ser uma ablação explícita, com orçamento e LR próprios.
        for module in [self.stem, *self.layers]:
            module.requires_grad_(train_encoder)
        self.decoders = nn.ModuleList(
            [block(768, 128), block(256, 64), block(128, 32), block(96, 32)]
        )
        self.head = nn.Conv2d(32, 4, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        # Estatísticas do encoder ficam estáveis em lotes pequenos; decoder usa GroupNorm.
        for module in [self.stem, *self.layers]:
            for layer in module.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, image):
        shape = image.shape[-2:]
        first = self.stem((image - self.mean) / self.std)
        features, x = [first], self.pool(first)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        x = features[-1]
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = decoder(torch.cat([x, skip], 1))
        return self.head(F.interpolate(x, size=shape, mode="bilinear", align_corners=False))


def build_model(config, initialize_pretrained=False):
    if config.architecture == "resnet18_unet":
        return ResNetUNet(
            pretrained=initialize_pretrained and config.pretrained,
            train_encoder=config.train_encoder,
        )
    if config.architecture == "unet":
        return UNet(config.base_channels)
    raise ValueError(f"Arquitetura desconhecida: {config.architecture}")

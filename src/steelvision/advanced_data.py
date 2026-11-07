"""Integridade, balanceamento e recortes nativos sem cruzar os splits históricos."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, Sampler

from .data import decode_rle, sha256_file


def records_for(manifest, split):
    return [r for r in json.loads(Path(manifest).read_text())["records"] if r["split"] == split]


def verify_integrity(data_dir, manifest, splits):
    """Verifica conteúdo, não só dimensões; nunca reescreve um manifesto existente."""
    payload = json.loads(Path(manifest).read_text())
    if sha256_file(Path(data_dir) / "train.csv") != payload["csv_sha256"]:
        raise ValueError("CSV alterado após a criação do manifesto")
    hashes = {}
    for record in payload["records"]:
        previous = hashes.setdefault(record["sha256"], record["split"])
        if previous != record["split"]:
            raise ValueError("Vazamento: hash presente em mais de um split")
        if record["split"] in splits:
            path = Path(data_dir) / "train_images" / record["image_id"]
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"Imagem alterada: {record['image_id']}")
    return {"manifest_sha256": sha256_file(manifest), "verified_splits": sorted(splits)}


def original_sample(data_dir, record):
    with Image.open(Path(data_dir) / "train_images" / record["image_id"]) as image:
        image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    masks = np.stack([decode_rle(r, record["height"], record["width"]) for r in record["rles"]])
    return image, masks


def training_statistics(manifest, max_pos_weight=20.0):
    """Todos os pesos são calculados exclusivamente nos registros de treino."""
    records = records_for(manifest, "train")
    if not records:
        raise ValueError("Treino vazio")
    present = np.array([[bool(rle) for rle in r["rles"]] for r in records])
    counts = present.sum(0)
    positive = np.zeros(4, dtype=np.float64)
    total = 0
    for record in records:
        total += record["height"] * record["width"]
        for k, rle in enumerate(record["rles"]):
            if rle:
                positive[k] += decode_rle(rle, record["height"], record["width"]).sum()
    # A raiz e o limite evitam multiplicadores extremos ao combinar amostragem e loss.
    pos_weight = np.clip(np.sqrt((total - positive) / np.maximum(positive, 1)), 1, max_pos_weight)
    inverse = np.sqrt(len(records) / np.maximum(counts, 1))
    sample_weights = np.where(present, inverse[None, :], 0).max(1)
    sample_weights[~present.any(1)] = 1.0
    sample_weights = np.minimum(sample_weights, 10.0)
    return {
        "source_split": "train",
        "positive_images": counts.tolist(),
        "positive_pixels": positive.astype(np.int64).tolist(),
        "pos_weight": pos_weight.tolist(),
        "sample_weights": sample_weights.tolist(),
        "formula": "sqrt(negative/positive) clipped; multilabel max sqrt(N/class_count)",
    }


class EpochSampler(Sampler):
    """Cada desenho inclui epoch e ocorrência: recortes determinísticos e distintos."""

    def __init__(self, weights, seed, epoch, balanced=True):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.seed, self.epoch, self.balanced = seed, epoch, balanced

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 100003)
        if self.balanced:
            order = torch.multinomial(self.weights, len(self.weights), True, generator=generator)
        else:
            order = torch.randperm(len(self.weights), generator=generator)
        return iter((int(index), draw, self.epoch) for draw, index in enumerate(order))

    def __len__(self):
        return len(self.weights)


class NativePatchDataset(Dataset):
    def __init__(
        self, data_dir, manifest, patch_size=(256, 400), seed=42, positive_crop_probability=0.5
    ):
        self.data_dir = Path(data_dir)
        self.records = records_for(manifest, "train")
        self.patch_size, self.seed = tuple(patch_size), seed
        self.positive_crop_probability = positive_crop_probability

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        index, draw, epoch = key if isinstance(key, tuple) else (key, key, 0)
        record = self.records[index]
        image, masks = original_sample(self.data_dir, record)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, draw, index]))
        ph, pw = self.patch_size
        h, w = image.shape[:2]
        if h < ph or w < pw:
            image = np.pad(image, ((0, max(0, ph - h)), (0, max(0, pw - w)), (0, 0)), mode="edge")
            masks = np.pad(masks, ((0, 0), (0, max(0, ph - h)), (0, max(0, pw - w))))
            h, w = image.shape[:2]
        top, left = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        available = np.flatnonzero(masks.reshape(4, -1).any(1))
        if len(available) and rng.random() < self.positive_crop_probability:
            k = int(rng.choice(available))
            points = np.argwhere(masks[k])
            cy, cx = points[int(rng.integers(len(points)))]
            top = int(np.clip(cy - int(rng.integers(ph)), 0, h - ph))
            left = int(np.clip(cx - int(rng.integers(pw)), 0, w - pw))
        image = Image.fromarray((image[top : top + ph, left : left + pw] * 255).astype(np.uint8))
        target = masks[:, top : top + ph, left : left + pw].copy()
        if rng.random() < 0.5:
            image, target = ImageOps.mirror(image), target[:, :, ::-1]
        if rng.random() < 0.5:
            image, target = ImageOps.flip(image), target[:, ::-1, :]
        image = ImageEnhance.Brightness(image).enhance(float(rng.uniform(0.8, 1.2)))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (
            torch.from_numpy(array.copy()),
            torch.from_numpy(target.copy().astype(np.float32)),
            record["image_id"],
        )

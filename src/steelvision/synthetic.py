"""Gera imagens geométricas para testar execução; não simula aço real."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .data import encode_rle


def generate_synthetic(output_dir, count=40, seed=42):
    output_dir = Path(output_dir)
    if (output_dir / "train.csv").exists():
        raise FileExistsError("Dataset já existe; escolha outro diretório")
    if count < 12:
        raise ValueError("Use pelo menos 12 exemplos sintéticos")
    image_dir = output_dir / "train_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    height, width = 64, 256
    rows = []
    colors = [(220, 60, 60), (60, 220, 60), (60, 60, 220), (230, 200, 50)]
    for i in range(count):
        image = rng.integers(35, 70, (height, width, 3), dtype=np.uint8)
        masks = np.zeros((4, height, width), dtype=np.uint8)
        # 20% de imagens normais; algumas imagens multiclasse.
        selected = [] if i % 5 == 0 else [i % 4]
        if i % 7 == 0 and selected:
            selected.append((i + 1) % 4)
        for k in selected:
            top, left = int(rng.integers(5, 35)), int(rng.integers(5, 210))
            masks[k, top : top + 15, left : left + 30] = 1
            image[masks[k] > 0] = colors[k]
        name = f"synthetic_{i:04d}.png"
        Image.fromarray(image).save(image_dir / name)
        for k in range(4):
            rows.append({"ImageId": name, "ClassId": k + 1, "EncodedPixels": encode_rle(masks[k])})
    with (output_dir / "train.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ImageId", "ClassId", "EncodedPixels"])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "SYNTHETIC.txt").write_text(
        "Dados geométricos artificiais para smoke test. Não são imagens Severstal ou aço real.\n"
    )
    return output_dir

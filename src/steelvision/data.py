"""Leitura Severstal, RLE por coluna e divisão antes de qualquer transformação."""

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

CLASSES = (1, 2, 3, 4)


def decode_rle(value, height, width):
    """Decodifica RLE Severstal: índices a partir de 1, ordem Fortran (colunas)."""
    mask = np.zeros(height * width, dtype=np.uint8)
    if value is None or not str(value).strip():
        return mask.reshape((height, width), order="F")
    try:
        runs = np.asarray(str(value).split(), dtype=np.int64)
    except ValueError as exc:
        raise ValueError("RLE deve conter inteiros") from exc
    if len(runs) % 2:
        raise ValueError("RLE deve conter pares início/comprimento")
    starts, lengths = runs[0::2] - 1, runs[1::2]
    if np.any(starts < 0) or np.any(lengths <= 0) or np.any(starts + lengths > mask.size):
        raise ValueError("RLE fora dos limites da imagem")
    for start, length in zip(starts, lengths):
        mask[start : start + length] = 1
    return mask.reshape((height, width), order="F")


def encode_rle(mask):
    """Inverso de decode_rle; usado também na demonstração sintética."""
    flat = (np.asarray(mask).ravel(order="F") > 0).astype(np.uint8)
    padded = np.concatenate([[0], flat, [0]])
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[0::2]
    return " ".join(map(str, changes))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_annotations(csv_path):
    """Aceita ImageId/ClassId e o formato legado ImageId_ClassId."""
    annotations = {}
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if "ImageId_ClassId" in row:
                image_id, class_id = row["ImageId_ClassId"].rsplit("_", 1)
            else:
                image_id, class_id = row["ImageId"], row["ClassId"]
            class_id = int(class_id)
            if class_id not in CLASSES:
                raise ValueError(f"Classe desconhecida: {class_id}")
            key = (image_id, class_id)
            if key in annotations:
                raise ValueError(f"Anotação duplicada: {key}")
            annotations[key] = (row.get("EncodedPixels") or "").strip()
    return annotations


def prepare_manifest(data_dir, output_dir, seed=42):
    """Divide imagens por grupos de hash, incluindo imagens sem defeitos."""
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    image_dir = data_dir / "train_images"
    annotations = read_annotations(data_dir / "train.csv")
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})
    if len(paths) < 3:
        raise ValueError("É necessário ter pelo menos três imagens")
    missing = {name for name, _ in annotations} - {p.name for p in paths}
    if missing:
        raise ValueError(f"Anotações sem imagem: {sorted(missing)[:5]}")
    records, groups = [], {}
    for path in paths:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        rles = [annotations.get((path.name, c), "") for c in CLASSES]
        # Validar os limites antes de iniciar um treinamento demorado.
        for rle in rles:
            decode_rle(rle, height, width)
        record = {
            "image_id": path.name,
            "sha256": sha256_file(path),
            "height": height,
            "width": width,
            "rles": rles,
        }
        records.append(record)
        groups.setdefault(record["sha256"], []).append(record)
    hashes = sorted(groups)
    if len(hashes) < 3:
        raise ValueError("É necessário ter três grupos de imagens distintas")
    # Conteúdo idêntico com anotações incompatíveis deve ser revisado.
    for group in groups.values():
        if len({tuple(r["rles"]) for r in group}) > 1:
            raise ValueError("Imagens idênticas possuem anotações conflitantes")
    random.Random(seed).shuffle(hashes)
    n_test = max(1, round(len(hashes) * 0.15))
    n_val = max(1, round(len(hashes) * 0.15))
    assignments = {}
    for i, digest in enumerate(hashes):
        assignments[digest] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for record in records:
        record["split"] = assignments[record["sha256"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "split_method": "seeded_exact_file_hash_groups_70_15_15",
        "csv_sha256": sha256_file(data_dir / "train.csv"),
        "records": records,
    }
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target


class SteelDataset(Dataset):
    """Máscaras independentes por classe; aumentos sincronizados só no treino."""

    def __init__(self, data_dir, manifest_path, split, size=(128, 800), augment=False):
        self.image_dir = Path(data_dir) / "train_images"
        manifest = json.loads(Path(manifest_path).read_text())
        self.records = [r for r in manifest["records"] if r["split"] == split]
        self.size = tuple(size)
        self.augment = augment
        if not self.records:
            raise ValueError(f"Divisão vazia: {split}")
        if augment and split != "train":
            raise ValueError("Aumentos permitidos somente no treino")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.image_dir / record["image_id"]) as original:
            image = original.convert("RGB")
        if image.size != (record["width"], record["height"]):
            raise ValueError("Dimensões alteradas após a criação do manifesto")
        masks = [
            Image.fromarray(decode_rle(rle, record["height"], record["width"]))
            for rle in record["rles"]
        ]
        if self.augment:
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
                masks = [ImageOps.mirror(m) for m in masks]
            if random.random() < 0.5:
                image = ImageOps.flip(image)
                masks = [ImageOps.flip(m) for m in masks]
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
        target_size = (self.size[1], self.size[0])
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        x = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        y = np.stack([np.asarray(m.resize(target_size, Image.Resampling.NEAREST)) for m in masks])
        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.astype(np.float32)),
            record["image_id"],
        )

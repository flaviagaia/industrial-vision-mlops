"""Treinamento, seleção por validação e avaliação única no teste reservado."""

import csv
import json
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import SteelDataset, sha256_file
from .metrics import SegmentationMetrics
from .model import UNet, segmentation_loss
from .visualize import plot_history, save_comparison


@dataclass
class Config:
    epochs: int = 10
    batch_size: int = 4
    height: int = 128
    width: int = 800
    base_channels: int = 16
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    patience: int = 4
    threshold: float = 0.5
    device: str = "auto"
    data_kind: str = "severstal"


def choose_device(value="auto"):
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def loader(data_dir, manifest, split, config, shuffle=False):
    dataset = SteelDataset(
        data_dir, manifest, split, (config.height, config.width), augment=shuffle
    )
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=config.device == "cuda",
    )


@torch.inference_mode()
def evaluate_loader(model, batches, device, threshold):
    model.eval()
    meter = SegmentationMetrics()
    loss_sum = count = 0
    for image, target, names in batches:
        image, target = image.to(device), target.to(device)
        logits = model(image)
        loss_sum += float(segmentation_loss(logits, target)) * len(names)
        count += len(names)
        meter.update(logits.sigmoid().cpu().numpy(), target.cpu().numpy(), names, threshold)
    summary = meter.summary()
    summary["loss"] = loss_sum / count
    return summary, meter.rows


def git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(data_dir, manifest, output_dir, config):
    if config.epochs < 1 or config.patience < 1:
        raise ValueError("epochs e patience devem ser positivos")
    if config.height < 8 or config.width < 8 or not 0 < config.threshold < 1:
        raise ValueError("Dimensões >= 8 e limiar entre 0 e 1 são obrigatórios")
    output_dir = Path(output_dir)
    if (output_dir / "best.pt").exists():
        raise FileExistsError("Use outro diretório para preservar a execução anterior")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    device = choose_device(config.device)
    model = UNet(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_batches = loader(data_dir, manifest, "train", config, shuffle=True)
    val_batches = loader(data_dir, manifest, "val", config)
    metadata = {
        "config": asdict(config),
        "device": str(device),
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": git_revision(),
        "manifest_sha256": sha256_file(manifest),
        "note": "Synthetic smoke test; not a Severstal result"
        if config.data_kind == "synthetic"
        else "Baseline; no production deployment claim",
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    # Copiar o manifesto fixa o split e as anotações usados nesta execução.
    (output_dir / "manifest.json").write_text(Path(manifest).read_text())
    best_score, best_loss, stale, history = -1.0, float("inf"), 0, []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = count = 0
        for image, target, _ in train_batches:
            image, target = image.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(image), target)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss não finita; verificar entradas e taxa de aprendizado")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * image.shape[0]
            count += image.shape[0]
        val, _ = evaluate_loader(model, val_batches, device, config.threshold)
        score = val["macro_dice"] if val["macro_dice"] is not None else 0.0
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "val_loss": val["loss"],
            "val_macro_dice": score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        # Val loss desempata Dice=0 no início sem consultar o teste.
        if score > best_score or (score == best_score and val["loss"] < best_loss):
            best_score, best_loss, stale = score, val["loss"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_metrics": val,
                    "manifest_sha256": metadata["manifest_sha256"],
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        if stale >= config.patience:
            break
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["best_val_macro_dice"] = best_score
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    plot_history(history, output_dir / "learning_curves.png")
    return output_dir / "best.pt"


def load_checkpoint(checkpoint_path, device="auto"):
    device = choose_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = Config(**checkpoint["config"])
    model = UNet(config.base_channels)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), config, checkpoint, device


@torch.inference_mode()
def evaluate(data_dir, manifest, checkpoint_path, output_dir, split="test", device="auto"):
    output_dir = Path(output_dir)
    if (output_dir / "metrics.json").exists():
        raise FileExistsError("Avaliação já existe; preserve o teste reservado")
    model, config, checkpoint, selected_device = load_checkpoint(checkpoint_path, device)
    if sha256_file(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("O manifesto difere daquele usado no treinamento")
    batches = loader(data_dir, manifest, split, config)
    summary, rows = evaluate_loader(model, batches, selected_device, config.threshold)
    summary.update(
        {
            "split": split,
            "threshold": config.threshold,
            "data_kind": config.data_kind,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evaluation_resolution": [config.height, config.width],
            "best_epoch": checkpoint["epoch"],
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    with (output_dir / "per_image_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Apenas erros efetivos, evitando mostrar vazios corretos como falhas.
    failures = sorted(
        [r for r in rows if r["fp_pixels"] + r["fn_pixels"] > 0],
        key=lambda r: (r["dice"], -r["fn_pixels"] - r["fp_pixels"]),
    )
    (output_dir / "worst_errors.json").write_text(json.dumps(failures[:30], indent=2))
    dataset = batches.dataset
    by_name = {r["image_id"]: i for i, r in enumerate(dataset.records)}
    chosen = list(dict.fromkeys(r["image_id"] for r in failures))[:6]
    if not chosen:
        chosen = [r["image_id"] for r in dataset.records[:3]]
    for n, name in enumerate(chosen):
        image, target, _ = dataset[by_name[name]]
        probabilities = model(image.unsqueeze(0).to(selected_device)).sigmoid()[0].cpu().numpy()
        save_comparison(
            image.numpy().transpose(1, 2, 0),
            target.numpy(),
            probabilities,
            output_dir / f"prediction_{n:02d}.png",
            f"{config.data_kind} | {split} | {name}",
            config.threshold,
        )
    lines = [
        "# Relatório de avaliação",
        "",
        f"Dados: **{config.data_kind}**. Divisão: **{split}**.",
        "",
        "Resultados em resolução de avaliação reduzida; não equivalem ao placar Kaggle.",
        "",
        "| Classe | Dice global | IoU | Recall por pixel | Dice imagens positivas |",
        "|---|---:|---:|---:|---:|",
    ]

    def fmt(x):
        return "n/a" if x is None else f"{x:.4f}"

    for item in summary["classes"]:
        lines.append(
            f"| {item['class_id']} | {fmt(item['dice'])} | {fmt(item['iou'])} | "
            f"{fmt(item['pixel_recall'])} | {fmt(item['positive_image_mean_dice'])} |"
        )
    lines += [
        "",
        "O checkpoint foi selecionado apenas na validação. Limiar fixo em 0,5.",
        "Máscaras vazias corretas não elevam o Dice global por classe.",
        "Consulte per_image_class.csv, worst_errors.json e prediction_*.png.",
    ]
    if config.data_kind == "synthetic":
        lines += [
            "",
            "**DEMONSTRAÇÃO SINTÉTICA: estas métricas não medem desempenho em aço real.**",
        ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


@torch.inference_mode()
def predict(image_path, checkpoint_path, output_dir, device="auto"):
    from PIL import Image

    from .data import encode_rle

    model, config, _, device = load_checkpoint(checkpoint_path, device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        original_size = image.size
        resized = image.convert("RGB").resize(
            (config.width, config.height), Image.Resampling.BILINEAR
        )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
    logits = model(tensor)
    # Restaurar probabilidades à resolução original antes de aplicar limiar.
    probabilities = torch.nn.functional.interpolate(
        logits.sigmoid(),
        size=(original_size[1], original_size[0]),
        mode="bilinear",
        align_corners=False,
    )
    masks = probabilities[0].cpu().numpy() >= config.threshold
    np.savez_compressed(output_dir / "prediction.npz", masks=masks)
    payload = {
        "image_id": Path(image_path).name,
        "width": original_size[0],
        "height": original_size[1],
        "threshold": config.threshold,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "classes": [{"class_id": k + 1, "encoded_pixels": encode_rle(masks[k])} for k in range(4)],
    }
    (output_dir / "prediction.json").write_text(json.dumps(payload, indent=2))
    display_prob = logits.sigmoid()[0].cpu().numpy()
    save_comparison(
        array,
        None,
        display_prob,
        output_dir / "prediction.png",
        "Inferência: sem anotação de referência",
        config.threshold,
    )
    return payload

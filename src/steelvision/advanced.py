"""Protocolo v2: treino balanceado, retomada, calibração e avaliação nativa."""

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .advanced_data import (
    EpochSampler,
    NativePatchDataset,
    original_sample,
    records_for,
    training_statistics,
    verify_integrity,
)
from .advanced_inference import ThresholdSearch, native_probabilities
from .data import encode_rle, sha256_file
from .engine import choose_device, seed_everything
from .metrics import SegmentationMetrics
from .model import segmentation_loss
from .transfer import build_model
from .visualize import plot_history, save_comparison


@dataclass
class AdvancedConfig:
    epochs: int = 6
    batch_size: int = 8
    patch_height: int = 256
    patch_width: int = 400
    architecture: str = "resnet18_unet"
    pretrained: bool = True
    train_encoder: bool = False
    base_channels: int = 16
    learning_rate: float = 0.0003
    seed: int = 42
    num_workers: int = 2
    patience: int = 4
    mixed_precision: bool = True
    balanced_sampling: bool = True
    weighted_loss: bool = True
    max_pos_weight: float = 20.0
    positive_crop_probability: float = 0.5
    device: str = "auto"
    data_kind: str = "severstal"


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32), *value[2:]))
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def load_model(checkpoint, device="auto"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = AdvancedConfig(**saved["config"])
    model = build_model(config, initialize_pretrained=False)
    model.load_state_dict(saved["model_state"])
    target = choose_device(device)
    return model.to(target).eval(), config, saved, target


def native_pass(model, config, data_dir, manifest, split, device, thresholds=0.5, search=None):
    meter = SegmentationMetrics()
    records = records_for(manifest, split)
    if not records:
        raise ValueError(f"Split vazio: {split}")
    started = time.perf_counter()
    for index, record in enumerate(records, 1):
        image, target = original_sample(data_dir, record)
        prob = native_probabilities(
            model,
            image,
            (config.patch_height, config.patch_width),
            device,
            config.batch_size,
            config.mixed_precision,
        )
        meter.update(prob[None], target[None], [record["image_id"]], thresholds)
        if search is not None:
            search.update(prob, target)
        if index % 400 == 0 or index == len(records):
            print(
                json.dumps(
                    {
                        "phase": f"{split}_native",
                        "images": index,
                        "total": len(records),
                        "seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    return meter


def train_advanced(data_dir, manifest, output_dir, config, resume=None):
    output = Path(output_dir)
    if config.epochs < 1 or min(config.patch_height, config.patch_width) < 32:
        raise ValueError("Épocas positivas e patches de pelo menos 32 pixels são obrigatórios")
    if (
        config.batch_size < 1
        or config.num_workers < 0
        or config.learning_rate <= 0
        or config.max_pos_weight < 1
    ):
        raise ValueError("Batch, workers, learning rate ou peso máximo inválido")
    if not 0 <= config.positive_crop_probability <= 1:
        raise ValueError("Probabilidade de recorte inválida")
    integrity = verify_integrity(data_dir, manifest, {"train", "val"})
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Execução existe; forneça resume=last.pt ou outra pasta")
    if resume is not None and not (output / "best.pt").exists():
        raise ValueError("Retome na pasta original, preservando best.pt e last.pt")
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    seed_everything(config.seed)
    statistics = training_statistics(manifest, config.max_pos_weight)
    dataset = NativePatchDataset(
        data_dir,
        manifest,
        (config.patch_height, config.patch_width),
        config.seed,
        config.positive_crop_probability,
    )
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if saved:
        if config.epochs < saved["epoch"]:
            raise ValueError("O orçamento não pode ser menor que a época já concluída")
        previous, current = saved["config"].copy(), asdict(config)
        previous.pop("epochs")
        current.pop("epochs")
        if previous != current or saved["manifest_sha256"] != integrity["manifest_sha256"]:
            raise ValueError("Configuração ou manifesto incompatível com a retomada")
        if saved["source_sha256"] != source_hash():
            raise ValueError("Código alterado: não é uma retomada exata")
    model = build_model(config, initialize_pretrained=not bool(saved)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pos_weight = (
        torch.tensor(statistics["pos_weight"], device=device).reshape(4, 1, 1)
        if config.weighted_loss
        else None
    )
    history, best, stale, initial_epoch, elapsed = [], -1.0, 0, 0, 0.0
    if saved:
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        history, best, stale = saved["history"], saved["best_score"], saved["stale"]
        initial_epoch, elapsed = saved["epoch"], saved["elapsed_seconds"]
        restore_rng(saved["rng_state"])
    metadata = {
        "protocol": "v2_native",
        "config": asdict(config),
        **integrity,
        "source_sha256": source_hash(),
        "torch": str(torch.__version__),
        "device": str(device),
        "amp_enabled": amp,
        "resume_from_epoch": initial_epoch,
        "historical_test_already_inspected": True,
        "threshold_during_selection": 0.5,
    }
    (output / "manifest.json").write_text(Path(manifest).read_text())
    (output / "training_statistics.json").write_text(json.dumps(statistics, indent=2))
    started = time.perf_counter()
    for epoch in range(initial_epoch + 1, config.epochs + 1):
        if stale >= config.patience:
            break
        sampler = EpochSampler(
            statistics["sample_weights"], config.seed, epoch, config.balanced_sampling
        )
        batches = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed + epoch),
        )
        model.train()
        loss_sum = count = 0
        epoch_start = time.perf_counter()
        for step, (image, target, _) in enumerate(batches, 1):
            image, target = (
                image.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = segmentation_loss(model(image), target, pos_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss não finita na época {epoch}, passo {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(image)
            count += len(image)
            if step % 100 == 0 or step == len(batches):
                print(
                    json.dumps(
                        {
                            "phase": "train_v2",
                            "epoch": epoch,
                            "step": step,
                            "steps": len(batches),
                            "seconds": round(time.perf_counter() - epoch_start, 1),
                        }
                    ),
                    flush=True,
                )
        validation = native_pass(model, config, data_dir, manifest, "val", device).summary()
        score = validation["macro_dice"] or 0.0
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "val_loss": None,
            "val_macro_dice": score,
        }
        history.append(row)
        if score > best:
            best, stale = score, 0
            atomic_save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "val_metrics": validation,
                    **integrity,
                    "source_sha256": source_hash(),
                    "thresholds": [0.5] * 4,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        atomic_save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "config": asdict(config),
                "epoch": epoch,
                "history": history,
                "best_score": best,
                "stale": stale,
                "rng_state": rng_state(),
                "elapsed_seconds": elapsed + time.perf_counter() - started,
                **integrity,
                "source_sha256": source_hash(),
            },
            output / "last.pt",
        )
        (output / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    metadata.update(
        {
            "epochs_completed": len(history),
            "best_val_macro_dice": best,
            "elapsed_seconds": elapsed + time.perf_counter() - started,
        }
    )
    (output / "run.json").write_text(json.dumps(metadata, indent=2))
    plot_history(history, output / "learning_curves.png")
    return output / "best.pt"


def calibrate(data_dir, manifest, checkpoint, output_dir):
    output = Path(output_dir)
    if (output / "calibration.json").exists():
        raise FileExistsError("Calibração existe; não sobrescrever")
    model, config, saved, device = load_model(checkpoint)
    integrity = verify_integrity(data_dir, manifest, {"val"})
    if integrity["manifest_sha256"] != saved["manifest_sha256"]:
        raise ValueError("Manifesto incompatível")
    search = ThresholdSearch()
    fixed = native_pass(model, config, data_dir, manifest, "val", device, search=search).summary()
    result = {
        **search.result(),
        "fixed_threshold_metrics": fixed,
        "checkpoint_sha256": sha256_file(checkpoint),
        **integrity,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration.json").write_text(json.dumps(result, indent=2))
    saved["thresholds"] = result["thresholds"]
    saved["calibration_sha256"] = sha256_file(output / "calibration.json")
    atomic_save(saved, output / "calibrated.pt")
    return output / "calibrated.pt"


def evaluate_advanced(data_dir, manifest, checkpoint, output_dir, split="test"):
    output = Path(output_dir)
    if (output / "metrics.json").exists():
        raise FileExistsError("Avaliação existe; preserve o resultado")
    model, config, saved, device = load_model(checkpoint)
    integrity = verify_integrity(data_dir, manifest, {split})
    if integrity["manifest_sha256"] != saved["manifest_sha256"]:
        raise ValueError("Manifesto incompatível")
    thresholds = saved.get("thresholds", [0.5] * 4)
    meter = native_pass(model, config, data_dir, manifest, split, device, thresholds)
    metrics = {
        **meter.summary(),
        **integrity,
        "data_kind": config.data_kind,
        "split": split,
        "evaluation_role": "historical_known_test" if split == "test" else "development",
        "resolution": "original",
        "postprocessing": "native_tiles_mean_probabilities_then_threshold",
        "thresholds": thresholds,
        "best_epoch": saved["epoch"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "calibration_sha256": saved.get("calibration_sha256"),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    with (output / "per_image_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(meter.rows[0]))
        writer.writeheader()
        writer.writerows(meter.rows)
    failures = sorted(
        [r for r in meter.rows if r["fp_pixels"] + r["fn_pixels"]], key=lambda r: r["dice"]
    )
    selected = list(dict.fromkeys(r["image_id"] for r in failures))[:4]
    lookup = {r["image_id"]: r for r in records_for(manifest, split)}
    for index, name in enumerate(selected):
        image, target = original_sample(data_dir, lookup[name])
        prob = native_probabilities(
            model,
            image,
            (config.patch_height, config.patch_width),
            device,
            config.batch_size,
            config.mixed_precision,
        )
        save_comparison(
            image,
            target,
            prob,
            output / f"prediction_{index:02d}.png",
            f"v2 | {split} | {name}",
            np.array(thresholds).reshape(4, 1, 1),
        )
    return metrics


@torch.inference_mode()
def predict_advanced(image_path, checkpoint, output_dir):
    model, config, saved, device = load_model(checkpoint)
    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    probabilities = native_probabilities(
        model,
        array,
        (config.patch_height, config.patch_width),
        device,
        config.batch_size,
        config.mixed_precision,
    )
    thresholds = np.array(saved.get("thresholds", [0.5] * 4)).reshape(4, 1, 1)
    masks = probabilities >= thresholds
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "prediction.npz", masks=masks)
    payload = {
        "image_id": Path(image_path).name,
        "resolution": "original",
        "thresholds": thresholds.ravel().tolist(),
        "checkpoint_sha256": sha256_file(checkpoint),
        "classes": [{"class_id": k + 1, "encoded_pixels": encode_rle(masks[k])} for k in range(4)],
    }
    (output / "prediction.json").write_text(json.dumps(payload, indent=2))
    save_comparison(
        array, None, probabilities, output / "prediction.png", "v2 inferência nativa", thresholds
    )
    return payload

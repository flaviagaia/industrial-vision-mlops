"""Análise exploratória, curvas e comparação visual de máscaras."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = np.array([[1.0, 0.75, 0.15], [0.7, 0.35, 1.0], [0.1, 0.85, 0.8], [1.0, 0.25, 0.4]])


def overlay(image, masks):
    result = image.copy()
    for channel, color in zip(masks, COLORS):
        selected = channel > 0
        result[selected] = 0.55 * result[selected] + 0.45 * color
    return np.clip(result, 0, 1)


def save_comparison(image, target, probabilities, output, title, threshold=0.5):
    arrays, labels = [image], ["Imagem"]
    if target is not None:
        arrays.append(overlay(image, target))
        labels.append("Máscara real")
    if probabilities is not None:
        arrays.append(overlay(image, probabilities >= threshold))
        labels.append("Predição")
    fig, axes = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 4), constrained_layout=True)
    for ax, array, label in zip(np.atleast_1d(axes), arrays, labels):
        ax.imshow(array)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title + " | classes: 1 dourado · 2 roxo · 3 ciano · 4 rosa", fontsize=10)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def explore(data_dir, manifest_path, output_dir, size=(128, 800)):
    from .data import SteelDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest["records"]
    summary = {
        "images": len(records),
        "unique_file_hashes": len({r["sha256"] for r in records}),
        "splits": {},
        "resolution_counts": {},
    }
    for record in records:
        key = f"{record['width']}x{record['height']}"
        summary["resolution_counts"][key] = summary["resolution_counts"].get(key, 0) + 1
    for split in ["train", "val", "test"]:
        selected = [r for r in records if r["split"] == split]
        summary["splits"][split] = {
            "images": len(selected),
            "no_defect": sum(not any(r["rles"]) for r in selected),
            "positive_images_per_class": {
                str(k + 1): sum(bool(r["rles"][k]) for r in selected) for k in range(4)
            },
        }
    (output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2))
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for j, split in enumerate(["train", "val", "test"]):
        counts = summary["splits"][split]["positive_images_per_class"]
        ax.bar(np.arange(4) + j * 0.25, [counts[str(k)] for k in range(1, 5)], 0.25, label=split)
    ax.set_xticks(np.arange(4) + 0.25, ["Classe 1", "Classe 2", "Classe 3", "Classe 4"])
    ax.set_ylabel("Imagens positivas (classes podem coexistir)")
    ax.legend()
    fig.savefig(output_dir / "class_distribution.png", dpi=140)
    plt.close(fig)
    dataset = SteelDataset(data_dir, manifest_path, "train", size=size)
    # EDA visual usa apenas treino; os exemplos incluem positivos e negativos quando disponíveis.
    candidates = sorted(
        range(len(dataset)), key=lambda i: -sum(bool(v) for v in dataset.records[i]["rles"])
    )
    chosen = candidates[:3]
    negatives = [i for i in candidates if not any(dataset.records[i]["rles"])]
    if negatives and negatives[0] not in chosen:
        chosen.append(negatives[0])
    for i in chosen:
        image, mask, name = dataset[i]
        save_comparison(
            image.numpy().transpose(1, 2, 0),
            mask.numpy(),
            None,
            output_dir / f"example_{i}.png",
            f"EDA treino: {name}",
        )
    return summary


def plot_history(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="treino")
    if any(r["val_loss"] is not None for r in history):
        axes[0].plot(
            [r["epoch"] for r in history], [r["val_loss"] for r in history], label="validação"
        )
    axes[0].set_title("BCE + soft Dice")
    axes[0].legend()
    axes[1].plot([r["epoch"] for r in history], [r["val_macro_dice"] for r in history])
    axes[1].set_title("Macro Dice de validação · limiar fixo 0,5")
    for ax in axes:
        ax.set_xlabel("Época")
    fig.savefig(output, dpi=140)
    plt.close(fig)

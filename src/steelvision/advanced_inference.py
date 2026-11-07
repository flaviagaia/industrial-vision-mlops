"""Mesmo mosaico de inferência nativa para validação, calibração, teste e uso isolado."""

import numpy as np
import torch
from torch.nn import functional as F


def starts(length, patch):
    return sorted(set([*range(0, max(length - patch, 0) + 1, patch), max(0, length - patch)]))


@torch.inference_mode()
def native_probabilities(model, image, patch_size, device, batch_size=4, amp=True):
    h, w = image.shape[:2]
    ph, pw = patch_size
    padded = np.pad(image, ((0, max(0, ph - h)), (0, max(0, pw - w)), (0, 0)), mode="edge")
    positions = [(y, x) for y in starts(padded.shape[0], ph) for x in starts(padded.shape[1], pw)]
    sums = np.zeros((4, *padded.shape[:2]), dtype=np.float32)
    counts = np.zeros(padded.shape[:2], dtype=np.float32)
    model.eval()
    for i in range(0, len(positions), batch_size):
        selected = positions[i : i + batch_size]
        tiles = np.stack([padded[y : y + ph, x : x + pw].transpose(2, 0, 1) for y, x in selected])
        tensor = torch.from_numpy(tiles.copy()).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(tensor)
        if logits.shape[-2:] != (ph, pw):
            logits = F.interpolate(
                logits.float(), size=(ph, pw), mode="bilinear", align_corners=False
            )
        probabilities = logits.float().sigmoid().cpu().numpy()
        for (y, x), prob in zip(selected, probabilities):
            sums[:, y : y + ph, x : x + pw] += prob
            counts[y : y + ph, x : x + pw] += 1
    if not counts.all():
        raise RuntimeError("Mosaico com pixels sem cobertura")
    return (sums / counts[None])[:, :h, :w]


class ThresholdSearch:
    """Contagens exatas por limiar sem guardar todas as probabilidades do dataset."""

    def __init__(self, grid=None):
        self.grid = np.asarray(
            grid if grid is not None else np.arange(0.05, 1.0, 0.05), dtype=np.float64
        )
        if self.grid.ndim != 1 or not len(self.grid) or np.any(np.diff(self.grid) <= 0):
            raise ValueError("Grade deve ser crescente e não vazia")
        self.positive = np.zeros((4, len(self.grid) + 1), dtype=np.int64)
        self.negative = self.positive.copy()

    def update(self, probabilities, target):
        for k in range(4):
            buckets = np.searchsorted(self.grid, probabilities[k].ravel(), side="right")
            truth = target[k].ravel() > 0
            self.positive[k] += np.bincount(buckets[truth], minlength=len(self.grid) + 1)
            self.negative[k] += np.bincount(buckets[~truth], minlength=len(self.grid) + 1)

    def result(self):
        output = []
        for k in range(4):
            tp = np.cumsum(self.positive[k][::-1])[::-1][1:]
            fp = np.cumsum(self.negative[k][::-1])[::-1][1:]
            fn = self.positive[k].sum() - tp
            dice = np.divide(
                2 * tp,
                2 * tp + fp + fn,
                out=np.zeros_like(tp, dtype=float),
                where=2 * tp + fp + fn > 0,
            )
            # Se não há positivos de referência, não ajustar um limiar sem suporte.
            best = int(np.flatnonzero(dice == dice.max())[-1]) if self.positive[k].sum() else None
            output.append(
                {
                    "class_id": k + 1,
                    "threshold": float(self.grid[best]) if best is not None else 0.5,
                    "validation_dice": float(dice[best]) if best is not None else None,
                    "positive_pixels": int(self.positive[k].sum()),
                }
            )
        return {
            "source_split": "val",
            "objective": "global Dice per class; ties prefer higher threshold",
            "grid": self.grid.tolist(),
            "classes": output,
            "thresholds": [r["threshold"] for r in output],
        }

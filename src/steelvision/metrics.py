"""Métricas globais por classe e Dice separado em imagens positivas."""

import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class SegmentationMetrics:
    def __init__(self):
        self.tp = np.zeros(4, dtype=np.int64)
        self.fp = self.tp.copy()
        self.fn = self.tp.copy()
        self.positive_dice = [[] for _ in range(4)]
        self.negative_images = self.tp.copy()
        self.false_positive_images = self.tp.copy()
        self.positive_images = self.tp.copy()
        self.detected_positive_images = self.tp.copy()
        self.rows = []

    def update(self, probabilities, target, image_ids, threshold=0.5):
        pred = np.asarray(probabilities) >= threshold
        true = np.asarray(target) > 0
        if pred.shape != true.shape or pred.ndim != 4 or pred.shape[1] != 4:
            raise ValueError("Esperado [batch, 4, altura, largura] para predição e alvo")
        for i, image_id in enumerate(image_ids):
            for k in range(4):
                p, t = pred[i, k], true[i, k]
                tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
                self.tp[k] += tp
                self.fp[k] += fp
                self.fn[k] += fn
                denom = 2 * tp + fp + fn
                dice = 2 * tp / denom if denom else 1.0
                if t.any():
                    self.positive_dice[k].append(dice)
                    self.positive_images[k] += 1
                    self.detected_positive_images[k] += int(p.any())
                else:
                    self.negative_images[k] += 1
                    self.false_positive_images[k] += int(p.any())
                self.rows.append(
                    {
                        "image_id": image_id,
                        "class_id": k + 1,
                        "dice": dice,
                        "tp_pixels": tp,
                        "fp_pixels": fp,
                        "fn_pixels": fn,
                        "true_positive_pixels": int(t.sum()),
                    }
                )

    def summary(self):
        classes = []
        for k in range(4):
            tp, fp, fn = int(self.tp[k]), int(self.fp[k]), int(self.fn[k])
            classes.append(
                {
                    "class_id": k + 1,
                    "dice": ratio(2 * tp, 2 * tp + fp + fn),
                    "iou": ratio(tp, tp + fp + fn),
                    "pixel_precision": ratio(tp, tp + fp),
                    "pixel_recall": ratio(tp, tp + fn),
                    "positive_image_mean_dice": float(np.mean(self.positive_dice[k]))
                    if self.positive_dice[k]
                    else None,
                    "positive_images": int(self.positive_images[k]),
                    "image_presence_recall": ratio(
                        self.detected_positive_images[k], self.positive_images[k]
                    ),
                    "negative_images": int(self.negative_images[k]),
                    "negative_image_fp_rate": ratio(
                        self.false_positive_images[k], self.negative_images[k]
                    ),
                    "tp_pixels": tp,
                    "fp_pixels": fp,
                    "fn_pixels": fn,
                }
            )
        valid = [c["dice"] for c in classes if c["dice"] is not None]
        return {
            "classes": classes,
            "macro_dice": float(np.mean(valid)) if valid else None,
            "valid_classes": len(valid),
            "conventions": "global pixel counts; absent global class -> null; "
            "per-image empty/empty Dice=1; positive-image Dice excludes empty targets; "
            "image presence means at least one predicted positive pixel",
        }

import numpy as np
import torch

from steelvision.metrics import SegmentationMetrics
from steelvision.model import UNet, segmentation_loss


def test_metric_counts_and_empty_convention():
    true = np.zeros((2, 4, 2, 2), dtype=np.float32)
    pred = true.copy()
    true[0, 0, 0, :] = 1
    pred[0, 0, 0, 0] = 1  # TP=1, FN=1
    pred[1, 0, 1, 0] = 1  # FP=1 em imagem negativa
    meter = SegmentationMetrics()
    meter.update(pred, true, ["a", "b"])
    result = meter.summary()
    c = result["classes"][0]
    assert c["dice"] == 0.5
    assert c["iou"] == 1 / 3
    assert c["pixel_recall"] == 0.5
    assert c["positive_image_mean_dice"] == 2 / 3
    assert c["negative_image_fp_rate"] == 1
    assert result["classes"][1]["dice"] is None
    assert result["valid_classes"] == 1


def test_model_preserves_odd_shape_and_can_backward():
    torch.set_num_threads(2)
    model = UNet(base_channels=4)
    image = torch.randn(2, 3, 33, 65)
    target = torch.zeros(2, 4, 33, 65)
    logits = model(image)
    assert logits.shape == target.shape
    loss = segmentation_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.head.weight.grad is not None


def test_empty_correct_images_do_not_inflate_global_dice():
    target = np.zeros((10, 4, 2, 2))
    target[0, 0, 0, 0] = 1
    meter = SegmentationMetrics()
    meter.update(np.zeros_like(target), target, [str(i) for i in range(10)])
    assert meter.summary()["classes"][0]["dice"] == 0.0

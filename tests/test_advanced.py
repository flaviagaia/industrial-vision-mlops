import json

import numpy as np
import pytest
import torch
from PIL import Image

from steelvision.advanced import (
    AdvancedConfig,
    calibrate,
    evaluate_advanced,
    predict_advanced,
    train_advanced,
)
from steelvision.advanced_data import (
    EpochSampler,
    NativePatchDataset,
    training_statistics,
    verify_integrity,
)
from steelvision.advanced_inference import ThresholdSearch, native_probabilities, starts
from steelvision.data import prepare_manifest
from steelvision.metrics import SegmentationMetrics
from steelvision.synthetic import generate_synthetic


def test_hash_detects_same_dimension_modification(tmp_path):
    data = generate_synthetic(tmp_path / "data", 16)
    manifest = prepare_manifest(data, tmp_path / "prepared")
    payload = json.loads(manifest.read_text())
    record = next(r for r in payload["records"] if r["split"] == "train")
    path = data / "train_images" / record["image_id"]
    with Image.open(path) as source:
        image = source.copy()
    image.putpixel((0, 0), (255, 255, 255))
    image.save(path)
    with pytest.raises(ValueError, match="Imagem alterada"):
        verify_integrity(data, manifest, {"train"})


def test_statistics_ignore_validation_and_test_labels(tmp_path):
    data = generate_synthetic(tmp_path / "data", 24)
    manifest = prepare_manifest(data, tmp_path / "prepared")
    before = training_statistics(manifest)
    payload = json.loads(manifest.read_text())
    for row in payload["records"]:
        if row["split"] != "train":
            row["rles"] = ["1 100"] * 4
    manifest.write_text(json.dumps(payload))
    assert before == training_statistics(manifest)
    assert min(before["pos_weight"]) >= 1 and max(before["pos_weight"]) <= 20


def test_stateless_crops_and_sampler_are_reproducible(tmp_path):
    data = generate_synthetic(tmp_path / "data", 24)
    manifest = prepare_manifest(data, tmp_path / "prepared")
    dataset = NativePatchDataset(data, manifest, (32, 64))
    a, b = dataset[(0, 5, 2)], dataset[(0, 5, 2)]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert a[0].shape == (3, 32, 64) and a[1].shape == (4, 32, 64)
    weights = training_statistics(manifest)["sample_weights"]
    assert list(EpochSampler(weights, 42, 3)) == list(EpochSampler(weights, 42, 3))
    assert list(EpochSampler(weights, 42, 3)) != list(EpochSampler(weights, 42, 4))


def test_native_tiles_cover_boundaries_and_preserve_pointwise_prediction():
    class Pointwise(torch.nn.Module):
        def forward(self, x):
            return x[:, :1].repeat(1, 4, 1, 1)

    rng = np.random.default_rng(9)
    image = rng.random((37, 109, 3), dtype=np.float32)
    expected = torch.from_numpy(image[:, :, 0]).sigmoid().numpy()
    result = native_probabilities(Pointwise(), image, (32, 40), torch.device("cpu"), amp=False)
    assert result.shape == (4, 37, 109)
    assert np.allclose(result, expected[None], atol=1e-6)
    assert starts(109, 40) == [0, 40, 69]


def test_threshold_search_matches_brute_force_and_class_vector():
    rng = np.random.default_rng(4)
    prob = rng.random((4, 7, 11), dtype=np.float32)
    truth = rng.integers(0, 2, prob.shape, dtype=np.uint8)
    prob[0, 0, 0] = 0.5
    search = ThresholdSearch([0.2, 0.5, 0.8])
    search.update(prob, truth)
    result = search.result()
    for k in range(4):
        candidates = []
        for threshold in [0.2, 0.5, 0.8]:
            p, t = prob[k] >= threshold, truth[k] > 0
            tp, fp, fn = (p & t).sum(), (p & ~t).sum(), (~p & t).sum()
            candidates.append(2 * tp / (2 * tp + fp + fn))
        best = max(range(3), key=lambda i: (candidates[i], i))
        assert result["thresholds"][k] == [0.2, 0.5, 0.8][best]
    meter = SegmentationMetrics()
    meter.update(prob[None], truth[None], ["a"], result["thresholds"])
    assert meter.summary()["macro_dice"] >= 0


def test_resume_matches_uninterrupted_training(tmp_path):
    torch.set_num_threads(2)
    data = generate_synthetic(tmp_path / "data", 16)
    manifest = prepare_manifest(data, tmp_path / "prepared")
    args = dict(
        architecture="unet",
        pretrained=False,
        base_channels=4,
        patch_height=32,
        patch_width=64,
        batch_size=4,
        num_workers=0,
        device="cpu",
        data_kind="synthetic",
    )
    full = train_advanced(data, manifest, tmp_path / "full", AdvancedConfig(epochs=2, **args))
    train_advanced(data, manifest, tmp_path / "resumed", AdvancedConfig(epochs=1, **args))
    resumed = train_advanced(
        data,
        manifest,
        tmp_path / "resumed",
        AdvancedConfig(epochs=2, **args),
        resume=tmp_path / "resumed/last.pt",
    )
    a = torch.load(tmp_path / "full/last.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    for key in a["model_state"]:
        assert torch.equal(a["model_state"][key], b["model_state"][key]), key
    assert a["history"] == b["history"]
    calibrated = calibrate(data, manifest, resumed, tmp_path / "calibration")
    metrics = evaluate_advanced(data, manifest, calibrated, tmp_path / "evaluation")
    assert metrics["resolution"] == "original"
    assert metrics["evaluation_role"] == "historical_known_test"
    image = next((data / "train_images").glob("*.png"))
    payload = predict_advanced(image, calibrated, tmp_path / "inference")
    assert payload["thresholds"] == metrics["thresholds"]
    with np.load(tmp_path / "inference/prediction.npz") as pred:
        assert pred["masks"].shape == (4, 64, 256)
    assert full.exists()
    with pytest.raises(ValueError, match="Configuração"):
        train_advanced(
            data,
            manifest,
            tmp_path / "resumed",
            AdvancedConfig(epochs=3, seed=99, **args),
            resume=tmp_path / "resumed/last.pt",
        )


def test_transfer_encoder_shape_without_network():
    pytest.importorskip("torchvision")
    config = AdvancedConfig(pretrained=False)
    from steelvision.transfer import build_model

    model = build_model(config, initialize_pretrained=False).eval()
    with torch.inference_mode():
        assert model(torch.zeros(1, 3, 64, 96)).shape == (1, 4, 64, 96)

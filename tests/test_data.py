import json

import numpy as np
import pytest
import torch

from steelvision.data import (
    SteelDataset,
    decode_rle,
    encode_rle,
    prepare_manifest,
    read_annotations,
)
from steelvision.synthetic import generate_synthetic


def test_rle_column_major_one_based():
    # Teste manual independente do encoder: 1=linha0,col0; 4=linha0,col1.
    expected = np.array([[1, 1], [1, 0], [0, 0]], dtype=np.uint8)
    assert np.array_equal(decode_rle("1 2 4 1", 3, 2), expected)
    assert encode_rle(expected) == "1 2 4 1"


@pytest.mark.parametrize("rle", ["0 2", "6 2", "1", "1 -1", "abc 1"])
def test_invalid_rle_rejected(rle):
    with pytest.raises(ValueError):
        decode_rle(rle, 3, 2)


def test_empty_and_boundary_rle():
    assert decode_rle("", 3, 2).sum() == 0
    assert decode_rle("6 1", 3, 2)[2, 1] == 1


def test_legacy_csv(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("ImageId_ClassId,EncodedPixels\nimage_a.jpg_2,1 2\n")
    assert read_annotations(path) == {("image_a.jpg", 2): "1 2"}


def test_split_reproducible_and_exact_duplicates_stay_together(tmp_path):
    root = generate_synthetic(tmp_path / "data", 24)
    source = root / "train_images/synthetic_0000.png"
    (root / "train_images/duplicate.png").write_bytes(source.read_bytes())
    a = prepare_manifest(root, tmp_path / "a", 42)
    b = prepare_manifest(root, tmp_path / "b", 42)
    assert a.read_bytes() == b.read_bytes()
    records = json.loads(a.read_text())["records"]
    for digest in {r["sha256"] for r in records}:
        assert len({r["split"] for r in records if r["sha256"] == digest}) == 1
    assert {r["split"] for r in records} == {"train", "val", "test"}
    assert next(r for r in records if r["image_id"] == "duplicate.png")["rles"] == [""] * 4


def test_multilabel_masks_and_missing_negative_rows(tmp_path):
    root = generate_synthetic(tmp_path / "data", 24)
    manifest = prepare_manifest(root, tmp_path / "prepared")
    dataset = SteelDataset(root, manifest, "train", size=(32, 128))
    image, mask, _ = dataset[0]
    assert image.shape == (3, 32, 128)
    assert mask.shape == (4, 32, 128)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert torch.isfinite(image).all()
    with pytest.raises(ValueError, match="somente"):
        SteelDataset(root, manifest, "test", augment=True)


def test_corrupted_image_rejected_at_prepare(tmp_path):
    root = generate_synthetic(tmp_path / "data", 12)
    (root / "train_images/broken.jpg").write_bytes(b"not an image")
    with pytest.raises((OSError, ValueError)):
        prepare_manifest(root, tmp_path / "prepared")


def test_duplicate_annotation_rejected(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("ImageId,ClassId,EncodedPixels\na.jpg,1,\na.jpg,1,\n")
    with pytest.raises(ValueError, match="duplicada"):
        read_annotations(path)


def test_flip_keeps_mask_and_image_aligned(tmp_path, monkeypatch):
    root = generate_synthetic(tmp_path / "data", 12)
    manifest = prepare_manifest(root, tmp_path / "prepared")
    plain = SteelDataset(root, manifest, "train", size=(64, 256))
    augmented = SteelDataset(root, manifest, "train", size=(64, 256), augment=True)
    monkeypatch.setattr("steelvision.data.random.random", lambda: 0.0)
    monkeypatch.setattr("steelvision.data.random.uniform", lambda a, b: 1.0)
    x, y, _ = plain[0]
    ax, ay, _ = augmented[0]
    assert torch.equal(ay, y.flip(-1, -2))
    assert torch.equal(ax, x.flip(-1, -2))


def test_conflicting_exact_duplicates_rejected(tmp_path):
    root = generate_synthetic(tmp_path / "data", 12)
    original = root / "train_images/synthetic_0001.png"
    (root / "train_images/duplicate.png").write_bytes(original.read_bytes())
    # duplicate tem rótulo vazio implícito; original tem defeito.
    with pytest.raises(ValueError, match="conflitantes"):
        prepare_manifest(root, tmp_path / "prepared")

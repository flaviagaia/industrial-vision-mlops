import json

import pytest
import torch

from steelvision.data import prepare_manifest
from steelvision.engine import Config, evaluate, predict, train
from steelvision.synthetic import generate_synthetic


def test_train_checkpoint_evaluate_and_predict(tmp_path):
    torch.set_num_threads(2)
    data = generate_synthetic(tmp_path / "data", 16)
    manifest = prepare_manifest(data, tmp_path / "prepared")
    config = Config(
        epochs=1, height=32, width=128, base_channels=4, device="cpu", data_kind="synthetic"
    )
    checkpoint = train(data, manifest, tmp_path / "run", config)
    summary = evaluate(data, manifest, checkpoint, tmp_path / "evaluation", device="cpu")
    assert summary["data_kind"] == "synthetic"
    assert summary["split"] == "test"
    assert (tmp_path / "evaluation/report.md").exists()
    assert list((tmp_path / "evaluation").glob("prediction_*.png"))
    image = next((data / "train_images").glob("*.png"))
    payload = predict(image, checkpoint, tmp_path / "prediction", device="cpu")
    assert payload["width"] == 256 and payload["height"] == 64
    assert len(payload["classes"]) == 4
    with pytest.raises(FileExistsError):
        train(data, manifest, tmp_path / "run", config)
    with pytest.raises(FileExistsError):
        evaluate(data, manifest, checkpoint, tmp_path / "evaluation", device="cpu")
    changed = json.loads(manifest.read_text())
    changed["seed"] = 99
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifesto"):
        evaluate(data, wrong, checkpoint, tmp_path / "wrong_evaluation", device="cpu")

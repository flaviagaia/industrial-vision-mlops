"""Notebook v2 independente, com protocolo congelado antes do teste histórico."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(value):
    cells.append(nbf.v4.new_markdown_cell(value))


def code(value):
    cells.append(nbf.v4.new_code_cell(value))


md("""# SteelVision v2 — classes raras, resolução nativa e retomada
**Flávia Gaia · laboratório 1 de 4 · PyTorch**

Esta versão implementa amostragem balanceada, BCE ponderada com pesos limitados, recortes nativos,
encoder ResNet18 pré-treinado opcional, checkpoint de retomada e calibração por classe somente na validação.
Validação, calibração, teste e inferência isolada usam o mesmo mosaico na resolução original.

**Protocolo:** preservar os splits históricos por hash. O teste foi examinado no baseline e será denominado
**teste histórico conhecido**, não um novo conjunto cego. Não ajustar configurações olhando seu resultado.
Uma comprovação externa independente exige novas imagens de outro lote/período, que não temos aqui.

O candidato combina mudanças; eventual ganho não pode ser atribuído isoladamente a uma delas.
As ablações devem usar o mesmo split e orçamento, escolhidas na validação antes de consultar o teste.

**Kaggle:** dados Severstal anexados, GPU e Internet ativadas para baixar os pesos oficiais ImageNet.
Não há fallback silencioso para pesos aleatórios. `PRETRAINED=False` permite execução sem download.
Não submeter automaticamente à competição.

[Baseline e resultados anteriores](https://github.com/flaviagaia/industrial-vision-mlops) ·
[Severstal](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data)
""")
md("""## 1. Configuração antes da execução
O orçamento de quatro épocas é um experimento inicial. O teste não participa da seleção do checkpoint
nem da busca de limiares. Para retomar: mantenha a mesma pasta e configuração e indique `RESUME_FROM`.
A retomada é exata na fronteira de época; falha no meio de uma época repete essa época inteira.
""")
code("""from pathlib import Path
USE_SYNTHETIC = False
RUN_NAME = "severstal_v2_native_resnet18_01"
EPOCHS = 4
BATCH_SIZE = 8
ARCHITECTURE = "resnet18_unet"
PRETRAINED = True
BALANCED_SAMPLING = True
WEIGHTED_LOSS = True
RESUME_FROM = None  # ex.: Path("/kaggle/working/.../run/last.pt")
DATA_DIR_OVERRIDE = None
BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
OUTPUT = BASE / "steelvision_artifacts" / RUN_NAME
PACKAGE_ROOT = BASE / "steelvision_v2_source"
PACKAGE = PACKAGE_ROOT / "steelvision"
PACKAGE.mkdir(parents=True, exist_ok=True)
""")
md("## 2. Código incorporado do repositório")
for path in sorted((ROOT / "src/steelvision").glob("*.py")):
    source = path.read_text()
    if "'''" in source:
        raise ValueError(path)
    md(f"### {path.name}")
    code(
        f"source = r'''{source}'''\n(PACKAGE / {path.name!r}).write_text(source, encoding='utf-8')"
    )
md("""## 3. Integridade e dados
Se o manifesto já existir, ele é reutilizado e verificado: não recalcular hashes após uma alteração
para ocultar a divergência. Nenhum recorte atravessa os splits de imagens de origem.
""")
code("""import sys, json
import torch
from PIL import Image as PILImage
sys.path.insert(0, str(PACKAGE_ROOT))
from steelvision.data import prepare_manifest
from steelvision.advanced import AdvancedConfig, train_advanced, calibrate, evaluate_advanced, predict_advanced
from steelvision.advanced_data import verify_integrity, training_statistics, records_for
from steelvision.synthetic import generate_synthetic

if USE_SYNTHETIC:
    DATA_DIR = OUTPUT / "synthetic_data"
    if not DATA_DIR.exists():
        generate_synthetic(DATA_DIR, count=16)
    torch.set_num_threads(2)
else:
    candidates = sorted(p.parent for p in Path("/kaggle/input").rglob("train.csv")
                        if (p.parent / "train_images").is_dir())
    if DATA_DIR_OVERRIDE is None and len(candidates) != 1:
        raise RuntimeError("Anexe os dados da competição ou configure DATA_DIR_OVERRIDE")
    DATA_DIR = Path(DATA_DIR_OVERRIDE) if DATA_DIR_OVERRIDE is not None else candidates[0]
manifest = OUTPUT / "prepared" / "manifest.json"
if not manifest.exists():
    manifest = prepare_manifest(DATA_DIR, manifest.parent, seed=42)
print(verify_integrity(DATA_DIR, manifest, {"train", "val"}))
statistics = training_statistics(manifest)
print(json.dumps({k:v for k,v in statistics.items() if k != "sample_weights"}, indent=2))
""")
md("""## 4. Treino com pesos e amostragem exclusivos do treino
Patches de 256 × 400 preservam a escala original. Metade dos recortes de imagens positivas tenta incluir
um pixel anotado; os demais são uniformes. Imagens normais continuam presentes.
O encoder pré-treinado recebe normalização ImageNet incorporada; suas estatísticas BatchNorm ficam
congeladas para lotes pequenos. A loss e as reduções espaciais permanecem em FP32 sob AMP.

`last.pt` salva modelo, otimizador, GradScaler, RNGs, histórico e critério de parada a cada época.
A retomada verifica manifesto, código e configuração. `best.pt` preserva o checkpoint de melhor
macro Dice de validação nativa com limiar fixo 0,5.
""")
code("""config = AdvancedConfig(
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    patch_height=32 if USE_SYNTHETIC else 256,
    patch_width=64 if USE_SYNTHETIC else 400,
    architecture="unet" if USE_SYNTHETIC else ARCHITECTURE,
    pretrained=False if USE_SYNTHETIC else PRETRAINED,
    base_channels=4 if USE_SYNTHETIC else 16,
    num_workers=0 if USE_SYNTHETIC else 2,
    device="cpu" if USE_SYNTHETIC else "auto",
    data_kind="synthetic" if USE_SYNTHETIC else "severstal",
    balanced_sampling=BALANCED_SAMPLING, weighted_loss=WEIGHTED_LOSS)
checkpoint = train_advanced(DATA_DIR, manifest, OUTPUT / "run", config, resume=RESUME_FROM)
print("Melhor checkpoint:", checkpoint)
with PILImage.open(OUTPUT / "run" / "learning_curves.png") as image:
    display(image.copy())
""")
md("""## 5. Calibração somente na validação
Buscar por classe o limiar de maior Dice global em grade 0,05…0,95. Empates preferem o limiar maior.
A busca acumula contagens sem guardar todas as probabilidades. Classes sem suporte positivo mantêm 0,5.
O modelo não é retreinado nesta etapa. Os limiares e o hash da calibração acompanham `calibrated.pt`.
""")
code("""deployment_checkpoint = calibrate(DATA_DIR, manifest, checkpoint, OUTPUT / "calibration")
print((OUTPUT / "calibration" / "calibration.json").read_text())
""")
md("""## 6. Avaliação uma vez no teste histórico
O caminho de inferência é o mesmo usado na validação: mosaico nativo, média nas sobreposições e limiares
congelados. Não comparar diretamente este Dice nativo com o Dice em resolução reduzida do baseline
como se a única diferença fosse o modelo. Uma comparação controlada requer reavaliar o baseline no
mesmo caminho nativo, sem recalibrar seus limiares no teste.

Este teste histórico já foi visto em outro experimento. Não chamá-lo de validação externa ou nova
estimativa independente. As próximas escolhas de modelo continuam limitadas à validação.
""")
code("""metrics = evaluate_advanced(DATA_DIR, manifest, deployment_checkpoint, OUTPUT / "historical_test")
print(json.dumps(metrics, indent=2))
for path in sorted((OUTPUT / "historical_test").glob("prediction_*.png")):
    with PILImage.open(path) as image:
        display(image.copy())
""")
md("## 7. Inferência isolada com os mesmos limiares e resolução")
code("""example = records_for(manifest, "test")[0]["image_id"]
result = predict_advanced(DATA_DIR / "train_images" / example, deployment_checkpoint, OUTPUT / "inference")
print("Limiares publicados:", result["thresholds"])
with PILImage.open(OUTPUT / "inference" / "prediction.png") as image:
    display(image.copy())
""")
md("""## 8. Artefatos
O ZIP é para armazenamento no Kaggle/local. Não enviar imagens, máscaras ou manifesto RLE para o GitHub.
Publicar ali apenas métricas, configurações e gráficos agregados. ONNX, incidentes e API/MLflow são os
laboratórios 2–4. Os posts no LinkedIn serão preparados após os quatro laboratórios, com evidências reais.
""")
code("""import zipfile
archive = BASE / f"steelvision_{RUN_NAME}.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and "synthetic_data" not in path.relative_to(OUTPUT).parts:
            bundle.write(path, path.relative_to(OUTPUT))
print("Artefatos:", archive)
""")
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.validate(nb)
nbf.write(nb, ROOT / "notebooks/02_severstal_improved.ipynb")
print("Notebook v2 gerado.")

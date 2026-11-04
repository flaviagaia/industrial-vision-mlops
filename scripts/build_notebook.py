"""Gera notebook Kaggle independente a partir do pacote local, sem divergência de código."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""# SteelVision — segmentação de defeitos em aço
**Flávia Gaia · PyTorch · U-Net · Visão Computacional industrial**

Este notebook implementa exploração, preparação, treino, avaliação por classe e análise de erros.
Não contém resultados pré-calculados em dados Severstal.

**Para executar no Kaggle:** aceite as regras da competição Severstal, adicione seus dados ao notebook
e selecione GPU nas configurações, se disponível. O pacote está incorporado abaixo; não é necessário
clonar repositório, baixar pesos ou ativar internet. Bibliotecas esperadas: PyTorch, NumPy, Pillow e Matplotlib.

O teste reservado será uma divisão do conjunto público anotado de treino. Os dados oficiais sem rótulos
da competição não são usados para calcular qualidade. O experimento real usa até dez épocas com parada antecipada na validação.
Isso não pressupõe convergência nem desempenho competitivo.

[Severstal](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data) ·
[U-Net](https://arxiv.org/abs/1505.04597)
""")
md("""## 1. Configuração
`USE_SYNTHETIC = False` usa Severstal. Altere para `True` apenas para testar o fluxo sem a competição.
Resultados sintéticos são identificados e não medem qualidade industrial.

Para repetir, altere `RUN_NAME`. Não sobrescreva uma execução. Aumente `EPOCHS` somente após validar o fluxo.
O limiar é fixo em 0,5. Não use resultados do teste para ajustar modelos ou limiares.
""")
code("""from pathlib import Path

USE_SYNTHETIC = False
EPOCHS = 10
RUN_NAME = "severstal_real_amp_01"
SEED = 42
BATCH_SIZE = 8
DATA_DIR_OVERRIDE = None  # opcional: Path("/caminho/para/severstal")

BASE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
OUTPUT = BASE / "steelvision_artifacts" / RUN_NAME
if OUTPUT.exists():
    raise FileExistsError(f"A execução {OUTPUT} já existe. Escolha outro RUN_NAME.")
PACKAGE_ROOT = BASE / "steelvision_notebook_source"
PACKAGE = PACKAGE_ROOT / "steelvision"
PACKAGE.mkdir(parents=True, exist_ok=True)
""")
md("""## 2. Código do projeto
As próximas células gravam o pacote localmente. O mesmo código é mantido nos módulos do repositório.
Você pode examinar as implementações de RLE, split, U-Net, métricas e treinamento aqui.
""")
for path in sorted((ROOT / "src/steelvision").glob("*.py")):
    source = path.read_text()
    if "'''" in source:
        raise ValueError(f"Delimitador conflitante: {path}")
    md(f"### `{path.name}`")
    code(
        f"source = r'''{source}'''\n(PACKAGE / {path.name!r}).write_text(source, encoding='utf-8')"
    )
md("## 3. Localização e validação dos dados")
code("""import json
import sys
import torch
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(PACKAGE_ROOT))
from steelvision.data import prepare_manifest
from steelvision.engine import Config, train, evaluate, predict, choose_device
from steelvision.synthetic import generate_synthetic
from steelvision.visualize import explore

if USE_SYNTHETIC:
    DATA_DIR = generate_synthetic(OUTPUT / "synthetic_data", seed=SEED)
    torch.set_num_threads(2)
    SIZE, CHANNELS, DEVICE = (64, 256), 8, "cpu"
else:
    if DATA_DIR_OVERRIDE is not None:
        DATA_DIR = Path(DATA_DIR_OVERRIDE)
    else:
        candidates = sorted(p.parent for p in Path("/kaggle/input").rglob("train.csv")
                            if (p.parent / "train_images").is_dir())
        if len(candidates) != 1:
            raise RuntimeError("Adicione os dados Severstal ao notebook ou configure DATA_DIR_OVERRIDE. "
                               f"Encontrados {len(candidates)} datasets compatíveis.")
        DATA_DIR = candidates[0]
    SIZE, CHANNELS, DEVICE = (128, 800), 16, "auto"

manifest = prepare_manifest(DATA_DIR, OUTPUT / "prepared", seed=SEED)
print("Dados:", DATA_DIR)
print("Modo:", "SINTÉTICO — não usar como métrica Severstal" if USE_SYNTHETIC else "Severstal")
print("Dispositivo:", choose_device(DEVICE))
print("PyTorch:", torch.__version__)
print("Manifesto:", manifest)
""")
md("""## 4. Exploração das classes e máscaras
A divisão ocorre antes dos aumentos. Duplicatas exatas por hash ficam no mesmo conjunto.
Sem identificadores de bobina/lote, isso não garante independência industrial. A divisão não é
estratificada; confira se há exemplos suficientes por classe em validação e teste.
As imagens normais são preservadas, inclusive quando não possuem linhas positivas no CSV.
""")
code("""summary = explore(DATA_DIR, manifest, OUTPUT / "eda", size=SIZE)
print(json.dumps(summary, indent=2, ensure_ascii=False))

def show_file(path, figsize=None):
    with PILImage.open(path) as image:
        display(image.copy())

show_file(OUTPUT / "eda" / "class_distribution.png", (9, 5))
for path in sorted((OUTPUT / "eda").glob("example_*.png"))[:3]:
    show_file(path)
""")
md("""## 5. Treinamento da U-Net
Baseline compacto treinado do zero, sem pesos externos. Os quatro canais são independentes:
as classes podem coexistir. A loss combina BCEWithLogits e soft Dice.

Imagem: bilinear. Máscara: nearest-neighbor. Aumentos sincronizados apenas no treino.
A resolução 128 × 800 reduz custo, mas pode apagar defeitos pequenos; métricas são calculadas nessa resolução.
O melhor checkpoint é selecionado por macro Dice de validação, com desempate pela loss.
""")
code("""config = Config(epochs=EPOCHS, batch_size=BATCH_SIZE, height=SIZE[0], width=SIZE[1],
                base_channels=CHANNELS, seed=SEED, device=DEVICE,
                num_workers=0 if USE_SYNTHETIC else 2, mixed_precision=True,
                data_kind="synthetic" if USE_SYNTHETIC else "severstal")
checkpoint = train(DATA_DIR, manifest, OUTPUT / "run", config)
show_file(OUTPUT / "run" / "learning_curves.png", (11, 5))
print("Checkpoint:", checkpoint)
""")
md("""## 6. Avaliação final no teste reservado
Execute após congelar o experimento. Não use estas métricas para ajustar o modelo.
Para estudar erros durante desenvolvimento, use `evaluate(..., split="val")` em outro diretório.

O Dice global por classe não é inflado por máscaras vazias corretas. Há também Dice apenas em imagens
positivas, recall por pixel e taxa de falso positivo em imagens negativas. Classes ausentes recebem `null`
quando a métrica não é definida. Esses números não reproduzem o placar oficial Kaggle.
""")
code("""metrics = evaluate(DATA_DIR, manifest, checkpoint, OUTPUT / "test", device=DEVICE)
print((OUTPUT / "test" / "report.md").read_text())
print(json.dumps(metrics, indent=2, ensure_ascii=False))
for path in sorted((OUTPUT / "test").glob("prediction_*.png"))[:4]:
    show_file(path)
""")
md("""## 7. Inferência em uma imagem
O exemplo usa uma imagem do próprio teste reservado. A predição não modifica o checkpoint.
As probabilidades são restauradas às dimensões originais antes do limiar; a saída é salva em NPZ e RLE.
Não há máscara de referência no painel de inferência isolada.
""")
code("""records = json.loads(Path(manifest).read_text())["records"]
example = next(record["image_id"] for record in records if record["split"] == "test")
result = predict(DATA_DIR / "train_images" / example, checkpoint, OUTPUT / "inference", DEVICE)
print("Imagem:", result["image_id"], "| dimensões:", result["height"], result["width"])
show_file(OUTPUT / "inference" / "prediction.png")
""")
md("""## 8. Salvar os resultados e próximos experimentos
O ZIP abaixo contém os artefatos da execução e o checkpoint, sem redistribuir as imagens de treino.
Use o painel de arquivos de saída do Kaggle para baixar.

Próximas hipóteses, avaliadas na validação: maior resolução, encoder pré-treinado, amostragem balanceada,
pesos por classe e limiares por classe. Congele as decisões antes de uma nova avaliação final independente.
ONNX, API de inferência, MLflow e monitoramento de drift são extensões posteriores.
""")
code("""import zipfile

archive = BASE / f"steelvision_{RUN_NAME}.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and "synthetic_data" not in path.relative_to(OUTPUT).parts:
            bundle.write(path, path.relative_to(OUTPUT))
print("Artefatos disponíveis:", archive)
""")
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.validate(nb)
path = ROOT / "notebooks/01_severstal_unet.ipynb"
nbf.write(nb, path)
print(path)

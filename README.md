# SteelVision — inspeção visual de defeitos em aço

Projeto de portfólio de **Flávia Gaia** para praticar segmentação de imagens industriais com PyTorch. O pipeline lê as anotações Severstal, separa os dados, treina uma U-Net compacta, seleciona o checkpoint na validação e produz métricas por classe e análise visual dos erros.

**Estado atual:** código implementado e fluxo validável com dados geométricos sintéticos. O treinamento e a avaliação no Severstal precisam ser executados após adicionar os dados da competição. Nenhuma métrica sintética representa desempenho em aço real. Este projeto não foi implantado em produção.

## Comece pelo Kaggle

1. Abra a [competição Severstal](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data), entre na sua conta e aceite as regras para acessar os dados.
2. Crie um notebook e importe `notebooks/01_severstal_unet.ipynb`.
3. Adicione os dados da competição ao notebook e selecione um acelerador GPU, se disponível.
4. Execute as células na ordem. O notebook contém o próprio código do pacote: não depende de clonar GitHub, baixar pesos ou instalar bibliotecas pela internet.
5. Comece com duas épocas para verificar a execução. Depois use uma nova pasta de saída e aumente o treinamento. Examine as curvas e a validação antes de consultar o teste reservado.

O notebook salva checkpoint, histórico, configuração, métricas e figuras em `/kaggle/working/steelvision_artifacts`. O dataset público de treino é dividido internamente em treino/validação/teste. As imagens oficiais de teste da competição não são usadas para medir qualidade, pois seus rótulos não estão disponíveis.

## Execução local

Python 3.10 ou superior; ambiente testado registrado em `requirements-tested.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Para testar tudo sem baixar a competição:

```bash
steelvision demo --out artifacts/synthetic_demo --epochs 2
```

Isso gera imagens geométricas, manifesto, EDA, treinamento, avaliação e inferência. Os arquivos são marcados como **synthetic**. Use outro diretório para repetir uma execução.

Para dados reais, organize:

```text
data/severstal/
├── train.csv
└── train_images/
    ├── imagem_1.jpg
    └── ...
```

Execute:

```bash
steelvision prepare --data data/severstal --out artifacts/prepared
steelvision eda --data data/severstal --manifest artifacts/prepared/manifest.json --out artifacts/eda
steelvision train --data data/severstal --manifest artifacts/prepared/manifest.json --out artifacts/run_01 --epochs 10
steelvision evaluate --data data/severstal --manifest artifacts/run_01/manifest.json --checkpoint artifacts/run_01/best.pt --out artifacts/test_01
steelvision predict --image data/severstal/train_images/IMAGEM.jpg --checkpoint artifacts/run_01/best.pt --out artifacts/prediction_01
```

`--device auto` escolhe CUDA, MPS ou CPU. Para CPU ou pouca memória, reduza `--batch-size` e `--base-channels`. O comando `predict` restaura as probabilidades à resolução original antes de aplicar o limiar e salva máscaras `.npz`, RLE JSON e uma visualização.

## Decisões técnicas

- **Segmentação multirrótulo:** quatro canais independentes. As classes podem coexistir na mesma imagem. Sigmoid e BCEWithLogitsLoss são usadas, em vez de softmax sobre classes mutuamente exclusivas.
- **U-Net compacta:** encoder/decoder com conexões de atalho e GroupNorm, treinada do zero. Não exige pesos externos. É um baseline, sem promessa de desempenho competitivo.
- **Loss:** BCE + soft Dice para combinar supervisão por pixel e sobreposição das máscaras. Os parâmetros são fixados antes do teste.
- **RLE:** índices começam em 1 e percorrem as colunas. Máscaras são decodificadas nas dimensões originais antes do redimensionamento.
- **Pré-processamento:** RGB em `[0, 1]`, resize bilinear da imagem e nearest-neighbor das máscaras. Padrão 128 × 800, mantendo a proporção das imagens Severstal de 256 × 1600. A redução pode apagar defeitos pequenos; essa limitação aparece no relatório.
- **Aumento de dados:** reflexões horizontal/vertical sincronizadas entre imagem e máscara e variação de brilho, apenas no treino. São hipóteses de baseline, a serem avaliadas na validação.
- **Split:** aproximadamente 70/15/15 por grupos de hash SHA-256 do arquivo. Arquivos idênticos não atravessam os conjuntos. A divisão não é estratificada; a EDA mostra a distribuição de cada classe em cada conjunto. Hash de arquivo não detecta imagens semelhantes ou a mesma imagem recomprimida. Sem metadados de bobina/lote/data, não há garantia de independência industrial; dados operacionais exigiriam divisão por lote ou período.
- **Seleção:** melhor macro Dice de validação, com desempate pela loss. Early stopping usa somente validação. Teste reservado não participa do treino ou da seleção do limiar.
- **Rastreabilidade:** configuração, versões de bibliotecas, seed, commit Git quando disponível, manifesto copiado e hashes de dados/checkpoint. Seeds reduzem variação; não garantem resultados idênticos entre hardwares.

## Como interpretar os resultados

| Métrica | O que responde |
|---|---|
| Dice global por classe | Qual a sobreposição agregada dos pixels previstos e reais? |
| IoU global por classe | Qual a interseção dividida pela união? |
| Recall por pixel | Que proporção dos pixels de defeito foi encontrada? |
| Dice médio em imagens positivas | Como o modelo se comporta sem o efeito de imagens vazias corretas? |
| Taxa de falso positivo em imagens negativas | Quantas imagens sem a classe receberam pelo menos um pixel previsto? |
| Recall de presença por imagem | Quantas imagens com a classe tiveram algum pixel previsto? Não mede localização correta. |

Uma classe globalmente ausente na referência **e** na predição recebe `null`, e não Dice perfeito. No CSV por imagem, máscara vazia dos dois lados recebe Dice 1; por isso essa média não é a métrica principal. Os relatórios não reproduzem o placar oficial Kaggle e usam a resolução reduzida informada em `metrics.json`.

Arquivos de uma execução:

```text
artifacts/
├── prepared/manifest.json
├── eda/data_summary.json e class_distribution.png
├── run_01/best.pt, run.json, manifest.json, history.json, learning_curves.png
└── test_01/metrics.json, report.md, per_image_class.csv,
           worst_errors.json e prediction_*.png
```

As figuras comparam entrada, máscara real e predição. Amarelo, roxo, ciano e rosa representam classes 1 a 4. Os identificadores numéricos são preservados sem atribuir nomes de defeitos não fornecidos nos dados.

## Verificação

```bash
pytest -q
ruff check src tests
```

Os testes cobrem a orientação do RLE com exemplo manual, limites inválidos, CSV legado, imagens corrompidas, duplicatas e vazamento entre splits, sincronização dos aumentos, convenções das métricas, gradientes e fluxo treino → checkpoint → teste → inferência.

## Escopo e próximos passos

Esta primeira entrega cobre exploração, preparação, treinamento, avaliação e inferência. Exportação ONNX, serviço de inferência, MLflow, drift e canário/rollback serão extensões posteriores. Não há alegação de produção, de vínculo com a Usiminas ou de resultados já obtidos no dataset real.

Dados e checkpoints não entram no Git. A licença MIT cobre o código deste repositório; dados Severstal continuam sujeitos às regras da competição. O notebook não redistribui as imagens.

## Referências

- [Dados e regras de acesso Severstal](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data)
- [U-Net — artigo original](https://arxiv.org/abs/1505.04597)
- [PyTorch Dataset e DataLoader](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

Veja `docs/GUIA_DE_ESTUDO.md` para o roteiro de prática e `docs/synthetic_demo/` para evidências da execução sintética.

# Protocolo v2

A versão 2 cobre todas as melhorias identificadas na revisão do laboratório 1. O baseline real,
seu checkpoint e suas métricas continuam preservados. As evidências sintéticas são testes técnicos,
não resultados de inspeção industrial.

## O que muda

| Componente | Implementação | Evidência técnica |
|---|---|---|
| Classes raras | Amostragem por presença multirrótulo e BCE com pesos limitados | Estatísticas calculadas somente no treino |
| Resolução | Recortes 256 × 400 sem reduzir a imagem inteira | Mosaico cobre toda a imagem, inclusive bordas |
| Encoder | ResNet18 com pesos ImageNet opcionais, normalização incorporada e BatchNorm congelada | Carregamento real dos pesos; teste de forma sem rede |
| Calibração | Um limiar por classe, ajustado apenas na validação | Busca por contagens exatas; teste contra força bruta |
| Avaliação | Resolução original e mesmo mosaico da inferência | Validação, calibração, teste e predição compartilham função |
| Retomada | `last.pt` atômico com modelo, otimizador, scaler, RNGs e histórico | Duas épocas contínuas equivalem a uma + retomada |
| Integridade | SHA-256 do CSV e das imagens antes de treino, calibração e avaliação | Detecta alteração mantendo as mesmas dimensões |
| Rastreabilidade | Configuração, hash do código, manifesto, checkpoint e calibração | Não depende de haver `.git` no Kaggle |

A loss usa FP32 para reduzir somas de pixels com segurança sob AMP. O encoder pré-treinado é
baixado da fonte oficial do Torchvision; não há fallback silencioso. Para retomar ou carregar um
checkpoint existente não é necessário baixar pesos novamente.

## Retomada

Use a mesma pasta de saída e configuração e passe `resume=OUTPUT / 'run/last.pt'` a
`train_advanced`. Pode aumentar `epochs`, mantendo as demais opções. A retomada ocorre na fronteira
da época. Uma interrupção durante a época repete a época inteira. Configuração, manifesto ou código
incompatíveis produzem erro. A igualdade exata foi testada em CPU; diferentes versões de CUDA,
bibliotecas ou hardware podem alterar resultados numéricos.

## Validação e teste histórico

Manter a seed 42 e os grupos históricos por hash: 8.798 imagens de treino, 1.885 de validação e
1.885 de teste. Não redistribuir as imagens entre conjuntos para procurar um placar melhor.
O teste anterior já foi examinado; os novos relatórios usam `historical_known_test`, não afirmam
validação externa independente. Uma confirmação independente exige dados novos, idealmente por lote,
período ou câmera, ainda não disponíveis neste projeto.

Selecionar o checkpoint pelo macro Dice nativo de validação com limiar fixo 0,5. Após o treino,
ajustar os quatro limiares no mesmo conjunto de validação, congelar `calibrated.pt` e executar o
teste histórico uma vez. A calibração pode ser otimista na validação; não é uma métrica de teste.
Não comparar o Dice nativo com o Dice reduzido do baseline como se fossem a mesma medição.

## Ablações propostas

O primeiro candidato v2 combina mudanças para testar o fluxo completo. Não atribuir eventual melhora
a um único componente. Para medir contribuições, executar, somente com validação, o mesmo orçamento:

1. U-Net compacta, recortes uniformes nativos, amostragem uniforme e loss sem peso.
2. Acrescentar apenas amostragem balanceada.
3. Acrescentar apenas ponderação da loss ao candidato anterior.
4. Acrescentar apenas recortes orientados por positivos.
5. Trocar o encoder/decoder por ResNet18 U-Net com inicialização aleatória.
6. Manter a arquitetura e ativar os pesos ImageNet.
7. No checkpoint selecionado, comparar limiar 0,5 e limiares calibrados na validação.

Os parâmetros `balanced_sampling`, `weighted_loss`, `positive_crop_probability`, `architecture` e
`pretrained` permitem reproduzir esses candidatos. Este plano não significa que todas as ablações
já foram executadas. Guardar cada execução em uma pasta diferente e não selecionar pelo teste.

## Publicação

O notebook `02_severstal_improved.ipynb` contém os módulos incorporados e executa preparação,
treino, calibração, avaliação histórica e inferência. Checkpoints e exemplos visuais ficam no
Kaggle/local. No GitHub, publicar apenas métricas e gráficos agregados dos dados Severstal.

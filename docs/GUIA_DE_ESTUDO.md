# Roteiro de prática

## 1. Entenda as anotações

Abra a EDA, selecione uma imagem e compare o RLE com a máscara. Explique por que a ordem é por coluna e o índice inicial é 1. Um erro aqui desloca a supervisão sem necessariamente impedir o treinamento.

## 2. Execute o baseline

Use duas épocas para checar a infraestrutura. Depois execute um treinamento com orçamento maior em outra pasta. Confira quantas imagens positivas de cada classe existem nos três conjuntos. Se houver classes ausentes ou escassas, revise a estratégia de divisão antes de comparar modelos; não procure a seed de melhor placar.

## 3. Leia as métricas

Compare Dice global e Dice em imagens positivas. Um modelo que prevê tudo como fundo pode parecer bom na acurácia por pixel, mas ter recall de defeitos próximo de zero. Observe também os falsos positivos em imagens normais.

## 4. Investigue os piores erros

Abra `worst_errors.json` e as figuras. Separe casos de defeitos pequenos, baixa visibilidade, predição excessiva e classes confundidas. Anote hipóteses para a próxima execução; o teste reservado não deve orientar iterações. Use a validação para análise iterativa.

## 5. Faça uma alteração de cada vez

Compare resolução, largura da U-Net e aumentos com o mesmo manifesto. Alterar o limiar requer seleção na validação e congelamento antes do teste. Não reivindique melhoria sem registrar a configuração e as métricas.

## 6. Explique o projeto em entrevista

O treino real foi concluído. Consulte `docs/severstal_real/report.md` e use apenas os números medidos:

> Desenvolvi um projeto de portfólio de segmentação de defeitos em aço com U-Net em PyTorch. Implementei decodificação RLE, divisão por grupos de imagens idênticas, aumentos sincronizados e avaliação por classe. Registrei as versões dos dados e do checkpoint e analisei falsos negativos e falsos positivos. O treinamento real foi executado no Kaggle; é um laboratório, não uma implantação industrial.

Nesta execução, foram usadas 12.568 imagens reais, com 10 épocas e seleção do checkpoint na época 7. O macro Dice de teste foi 0,2443, com recall zero nas classes 1 e 2. Explique essa limitação e diferencie o laboratório de uma implantação industrial. ONNX e sustentação em produção não fazem parte desta entrega.

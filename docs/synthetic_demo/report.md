# Relatório de avaliação

Dados: **synthetic**. Divisão: **test**.

Resultados em resolução de avaliação reduzida; não equivalem ao placar Kaggle.

| Classe | Dice global | IoU | Recall por pixel | Dice imagens positivas |
|---|---:|---:|---:|---:|
| 1 | 0.7916 | 0.6550 | 1.0000 | 0.8099 |
| 2 | 0.8697 | 0.7694 | 0.8156 | 0.8951 |
| 3 | 0.0181 | 0.0092 | 1.0000 | 0.0535 |
| 4 | 0.0000 | 0.0000 | n/a | n/a |

O checkpoint foi selecionado apenas na validação. Limiar fixo em 0,5.
Máscaras vazias corretas não elevam o Dice global por classe.
Consulte per_image_class.csv, worst_errors.json e prediction_*.png.

**DEMONSTRAÇÃO SINTÉTICA: estas métricas não medem desempenho em aço real.**
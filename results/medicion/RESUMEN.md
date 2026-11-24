# Medición de rendimiento

Generado por `qcclass medir --resumen` a partir de los JSON de esta carpeta. Modelo de referencia (GINEConv), protocolo `aleatorio`, conjunto train. Las cifras de época son las de la última época medida (caché de disco caliente si hay dos o más).

| Equipo | Dispositivo | GPU | Presupuesto | Workers | Lotes/época | s/época | Espera datos | Nodos/s | Grafos/s | Pérdida 1→N | Mem. GPU máx. (MB) | Útil al inicio (MB) | RAM máx. (MB) | Temp. GPU máx./media | Peor caso | Mayor presupuesto (barrido) | 100 épocas (h) | Optuna 20 / 50 (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| colab-t4 | cuda | Tesla T4 | 100,000 | 0 | 71 | 12.0 | 69.1 % | 571,333 | 530.5 | 2.893 → 0.991 | 527.4 | 14,808 | 1,584 | 49.0 / 49.0 | 19,937 nodos, ok | 400,000 | 0.33 | 26.7 / 66.7 |
| colab-t4 | cuda + AMP | Tesla T4 | 100,000 | 0 | 71 | 11.3 | 70.3 % | 604,732 | 561.5 | 2.956 → 1.023 | 277.3 | 14,808 | 1,714 | — | 19,937 nodos, ok | 400,000 | 0.31 | 25.2 / 63.0 |
| m3pro | cpu | — | 100,000 | 4 | 72 | 7.41 | 1.3 % | 925,894 | 859.7 | 2.854 → 0.969 | — | — | 2,301 | — | 19,937 nodos, ok | — | 0.21 | 16.5 / 41.1 |
| m3pro | mps | Apple MPS (Apple M3 Pro) | 100,000 | 0 | 71 | 15.4 | 25.2 % | 445,474 | 413.6 | 2.884 → 0.969 | 1,203 | — | 1,776 | — | 19,937 nodos, ok | 400,000 | 0.43 | 34.2 / 85.5 |
| rtx3050-w4 | cuda + AMP | NVIDIA GeForce RTX 3050 Laptop GPU | 100,000 | 4 | 72 | 3.68 | 11.5 % | 1,864,378 | 1,731 | 2.896 → 0.956 | 275.5 | 3,300 | 1,867 | — | 19,937 nodos, ok | — | 0.1 | 8.2 / 20.4 |
| rtx3050 | cuda | NVIDIA GeForce RTX 3050 Laptop GPU | 100,000 | 0 | 71 | 8.28 | 57.5 % | 827,790 | 768.6 | 2.890 → 0.985 | 527.4 | 3,300 | 1,618 | — | 19,937 nodos, ok | 400,000 | 0.23 | 18.4 / 46.0 |
| rtx3050 | cuda + AMP | NVIDIA GeForce RTX 3050 Laptop GPU | 100,000 | 0 | 71 | 7.9 | 61.4 % | 867,678 | 805.7 | 2.957 → 1.018 | 277.3 | 3,300 | 1,830 | — | 19,937 nodos, ok | 400,000 | 0.22 | 17.6 / 43.9 |

Versión del dataset: dataset-v1 (b820a9d62674).

## Barrido de presupuestos

**colab-t4 (cuda)**

| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |
|---|---|---|---|
| 25,000 | ok | 544,850 | 134.7 |
| 50,000 | ok | 574,136 | 264.7 |
| 100,000 | ok | 582,797 | 526.7 |
| 200,000 | ok | 636,880 | 1,048 |
| 400,000 | ok | 644,379 | 2,094 |

**colab-t4 (cuda)**

| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |
|---|---|---|---|
| 25,000 | ok | 440,812 | 72.6 |
| 50,000 | ok | 607,158 | 139.6 |
| 100,000 | ok | 593,245 | 277.1 |
| 200,000 | ok | 673,446 | 549.8 |
| 400,000 | ok | 725,557 | 1,094 |

**m3pro (mps)**

| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |
|---|---|---|---|
| 25,000 | ok | 63,443 | 255.3 |
| 50,000 | ok | 427,644 | 1,128 |
| 100,000 | ok | 459,410 | 1,145 |
| 200,000 | ok | 449,322 | 2,184 |
| 400,000 | ok | 501,661 | 4,224 |

**rtx3050 (cuda)**

| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |
|---|---|---|---|
| 25,000 | ok | 651,496 | 134.7 |
| 50,000 | ok | 773,294 | 264.7 |
| 100,000 | ok | 858,865 | 526.7 |
| 200,000 | ok | 875,542 | 1,048 |
| 400,000 | ok | 725,114 | 2,094 |

**rtx3050 (cuda)**

| Presupuesto | Estado | Nodos/s | Mem. GPU máx. (MB) |
|---|---|---|---|
| 25,000 | ok | 536,963 | 72.6 |
| 50,000 | ok | 814,513 | 139.6 |
| 100,000 | ok | 919,987 | 277.1 |
| 200,000 | ok | 956,806 | 549.8 |
| 400,000 | ok | 1,020,642 | 1,094 |

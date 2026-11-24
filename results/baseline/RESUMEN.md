# Línea base (fase 10)

Dataset dataset-v1.1 (0f5f28fcf120), commit `e9856d7`, equipo `m3pro`. Validación cruzada de 5 bloques sobre train + val de `aleatorio`; objetivo de Optuna = F1 macro medio de los bloques. Columnas «OOF»: predicciones fuera de bloque del mejor intento, con intervalos del 95 % por bootstrap de grupos. **Sin test.**

## Objetivo: algoritmo

| Peldaño | Clasificador | F1 macro CV (%) | F1 macro OOF [IC 95 %] | Exactitud bal. | Exactitud | Top-3 | MCC | Intentos | Búsqueda (min) |
|---|---|---|---|---|---|---|---|---|---|
| e7 | hist_gradient_boosting | 56.1 ± 1.1 | 56.2 [54.5, 57.7] | 55.6 | 67.0 | 89.8 | 64.2 | complete 34, pruned 16 | 59.0 |
| e7 | random_forest | 57.3 ± 0.6 | 57.5 [55.7, 58.9] | 57.8 | 67.5 | 90.3 | 64.9 | complete 35, pruned 15 | 1.8 |
| e7 | regresion_logistica | 39.0 ± 0.7 | 39.0 [37.8, 40.1] | 45.7 | 49.8 | 80.3 | 46.6 | complete 32, pruned 18 | 1.2 |
| f3 | hist_gradient_boosting | 83.9 ± 2.7 | 83.9 [82.5, 85.1] | 83.9 | 85.1 | 98.2 | 83.9 | complete 25, pruned 25 | 21.4 |
| f3 | random_forest | 84.8 ± 1.8 | 84.8 [83.4, 85.9] | 85.0 | 86.5 | 98.4 | 85.4 | complete 36, pruned 14 | 2.3 |
| f3 | regresion_logistica | 82.7 ± 1.0 | 82.7 [81.5, 83.8] | 83.3 | 84.8 | 98.5 | 83.7 | complete 31, pruned 19 | 2.3 |
| wl | hist_gradient_boosting | 97.7 ± 0.7 | 97.7 [97.0, 98.3] | 97.7 | 98.5 | 99.6 | 98.4 | complete 40, pruned 10, fail 1 | 117.1 |
| wl | random_forest | 97.1 ± 0.9 | 97.1 [96.3, 97.8] | 97.2 | 98.3 | 99.6 | 98.2 | complete 31, pruned 19 | 2.1 |
| wl | regresion_logistica | 96.9 ± 0.5 | 96.9 [96.2, 97.5] | 97.0 | 98.2 | 99.6 | 98.0 | complete 27, pruned 23 | 18.3 |

## Objetivo: familia

| Peldaño | Clasificador | F1 macro CV (%) | F1 macro OOF [IC 95 %] | Exactitud bal. | Exactitud | Top-3 | MCC | Intentos | Búsqueda (min) |
|---|---|---|---|---|---|---|---|---|---|
| e7 | hist_gradient_boosting | 74.8 ± 1.7 | 74.9 [72.6, 76.8] | 74.6 | 85.6 | 98.5 | 78.5 | complete 26, pruned 24 | 16.1 |
| e7 | random_forest | 75.7 ± 2.3 | 75.7 [73.4, 77.6] | 77.3 | 85.1 | 98.4 | 78.3 | complete 31, pruned 19 | 1.6 |
| e7 | regresion_logistica | 42.2 ± 1.9 | 42.3 [40.3, 44.0] | 53.6 | 54.7 | 91.0 | 40.3 | complete 41, pruned 9 | 0.8 |
| f3 | hist_gradient_boosting | 96.7 ± 1.2 | 96.7 [95.6, 97.6] | 96.7 | 98.3 | 99.6 | 97.5 | complete 27, pruned 23 | 7.3 |
| f3 | random_forest | 96.1 ± 1.3 | 96.2 [94.9, 97.2] | 96.6 | 98.0 | 99.7 | 97.0 | complete 35, pruned 15 | 1.9 |
| f3 | regresion_logistica | 89.4 ± 1.2 | 89.4 [87.6, 90.9] | 92.7 | 93.3 | 98.6 | 90.2 | complete 35, pruned 15 | 0.9 |
| wl | hist_gradient_boosting | 98.8 ± 0.7 | 98.8 [98.2, 99.4] | 98.8 | 99.5 | 99.8 | 99.3 | pruned 31, complete 19 | 41.9 |
| wl | random_forest | 98.6 ± 0.8 | 98.6 [97.8, 99.2] | 98.4 | 99.3 | 99.8 | 99.0 | complete 30, pruned 20 | 1.8 |
| wl | regresion_logistica | 98.8 ± 0.5 | 98.8 [98.1, 99.3] | 98.6 | 99.4 | 99.8 | 99.1 | pruned 27, complete 23 | 3.4 |

## Finalistas (mejor clasificador de cada peldaño, F1 macro de algoritmo)

- **e7**: `random_forest`, F1 macro CV 57.3 %, OOF 57.5 % [55.7, 58.9]. Parámetros: `{"n_estimators": 200, "max_depth": null, "max_features": 1.0, "min_samples_leaf": 1}`
- **f3**: `random_forest`, F1 macro CV 84.8 %, OOF 84.8 % [83.4, 85.9]. Parámetros: `{"n_estimators": 300, "max_depth": 16, "max_features": 0.5, "min_samples_leaf": 1}`
- **wl**: `hist_gradient_boosting`, F1 macro CV 97.7 %, OOF 97.7 % [97.0, 98.3]. Parámetros: `{"h": 2, "learning_rate": 0.09999227353112966, "l2_regularization": 3.337201737416827e-05, "min_circuitos": 5, "min_samples_leaf": 33, "columnas_wl": 1024, "max_leaf_nodes": 24, "max_iter": 250}`

## Importancia por permutación: e7

| Característica | Caída de F1 macro (puntos) | Desv. entre bloques |
|---|---|---|
| profundidad | 47.25 | 0.34 |
| n_puertas | 41.39 | 0.68 |
| n_qubits | 30.6 | 0.88 |

## Importancia por permutación: f3

| Característica | Caída de F1 macro (puntos) | Desv. entre bloques |
|---|---|---|
| grafo_densidad | 12.44 | 1.09 |
| grafo_grado_medio | 9.99 | 2.06 |
| prop_sx | 8.91 | 0.93 |
| cx_por_capa_media | 8.17 | 0.16 |
| prop_dos_cubits | 6.23 | 0.6 |
| prop_cx | 5.63 | 1.02 |
| prop_rz | 4.97 | 0.46 |
| prop_x | 4.67 | 1.59 |
| cx_por_capa_max | 3.83 | 0.52 |
| n_x | 3.83 | 0.37 |
| profundidad | 3.63 | 0.38 |
| grafo_diametro | 1.11 | 0.48 |
| grafo_componentes | 1.03 | 0.37 |
| n_sx | 0.17 | 0.32 |
| n_cx | 0.09 | 0.22 |

## Importancia por permutación: wl

| Característica | Caída de F1 macro (puntos) | Desv. entre bloques |
|---|---|---|
| wl#857971 | 3.95 | 0.41 |
| cx | 3.04 | 1.66 |
| rz(-8pi/16) | 2.92 | 0.42 |
| wl#322033 | 2.59 | 0.78 |
| wl#302667 | 1.03 | 0.42 |
| wl#505292 | 0.94 | 0.11 |
| wl#321721 | 0.65 | 0.22 |
| wl#879078 | 0.44 | 0.27 |
| sx | 0.36 | 0.24 |
| rz(-4pi/16) | 0.34 | 0.18 |
| wl#173268 | 0.26 | 0.19 |
| rz(otro) | 0.23 | 0.24 |
| wl#856826 | 0.21 | 0.13 |
| wl#237624 | 0.19 | 0.13 |
| wl#609946 | 0.19 | 0.14 |

Tiempo total de búsqueda: 5.0 h.

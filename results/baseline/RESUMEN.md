# Línea base (fase 10)

Dataset dataset-v1.1 (0f5f28fcf120), commit `75be38c`, equipo `m3pro`. Validación cruzada de 5 bloques sobre train + val de `aleatorio`; objetivo de Optuna = F1 macro medio de los bloques. Columnas «OOF»: predicciones fuera de bloque del mejor intento, con intervalos del 95 % por bootstrap de grupos. **Sin test.**

## Objetivo: algoritmo

| Peldaño | Clasificador | F1 macro CV (%) | F1 macro OOF [IC 95 %] | Exactitud bal. | Exactitud | Top-3 | MCC | Rev. | Intentos | Búsqueda (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| e7 | hist_gradient_boosting | 56.3 ± 0.7 | 56.4 [54.8, 57.8] | 55.9 | 67.3 | 88.7 | 64.6 | 2 | complete 33, fail 1, pruned 17 | 69.1 |
| e7 | random_forest | 57.3 ± 0.6 | 57.5 [55.7, 58.9] | 57.8 | 67.5 | 90.3 | 64.9 | 1 | complete 35, pruned 15 | 1.8 |
| e7 | regresion_logistica | 39.0 ± 0.7 | 39.1 [37.8, 40.1] | 45.7 | 49.8 | 80.3 | 46.6 | 2 | complete 26, pruned 24, running 1 | 1.2 |
| f3 | hist_gradient_boosting | 83.9 ± 2.7 | 83.9 [82.5, 85.1] | 83.9 | 85.1 | 98.2 | 83.9 | 1 | complete 25, pruned 25 | 21.4 |
| f3 | random_forest | 84.8 ± 1.8 | 84.8 [83.4, 85.9] | 85.0 | 86.5 | 98.4 | 85.4 | 1 | complete 36, pruned 14 | 2.3 |
| f3 | regresion_logistica | 83.2 ± 0.7 | 83.3 [82.0, 84.3] | 83.7 | 85.4 | 98.5 | 84.4 | 2 | complete 24, pruned 26 | 3.7 |
| wl | hist_gradient_boosting | 97.7 ± 0.7 | 97.8 [97.1, 98.3] | 97.8 | 98.5 | 99.6 | 98.4 | 2 | complete 27, pruned 23 | 142.8 |
| wl | random_forest | 97.1 ± 1.0 | 97.1 [96.3, 97.7] | 97.2 | 98.2 | 99.6 | 98.1 | 2 | complete 28, pruned 22 | 2.6 |
| wl | regresion_logistica | 97.0 ± 0.3 | 97.0 [96.3, 97.6] | 97.1 | 98.3 | 99.5 | 98.1 | 2 | complete 21, pruned 29 | 12.0 |

## Objetivo: familia

| Peldaño | Clasificador | F1 macro CV (%) | F1 macro OOF [IC 95 %] | Exactitud bal. | Exactitud | Top-3 | MCC | Rev. | Intentos | Búsqueda (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| e7 | hist_gradient_boosting | 74.8 ± 1.7 | 74.9 [72.6, 76.8] | 74.6 | 85.6 | 98.5 | 78.5 | 1 | complete 26, pruned 24 | 16.1 |
| e7 | random_forest | 75.7 ± 2.3 | 75.7 [73.4, 77.6] | 77.3 | 85.1 | 98.4 | 78.3 | 1 | complete 31, pruned 19 | 1.6 |
| e7 | regresion_logistica | 42.2 ± 1.9 | 42.3 [40.3, 44.0] | 53.6 | 54.7 | 91.0 | 40.3 | 1 | complete 41, pruned 9 | 0.8 |
| f3 | hist_gradient_boosting | 96.6 ± 1.3 | 96.6 [95.4, 97.5] | 96.4 | 98.3 | 99.7 | 97.5 | 2 | complete 18, pruned 32 | 12.2 |
| f3 | random_forest | 96.1 ± 1.3 | 96.2 [94.9, 97.2] | 96.6 | 98.0 | 99.7 | 97.0 | 1 | complete 35, pruned 15 | 1.9 |
| f3 | regresion_logistica | 89.5 ± 1.2 | 89.5 [87.8, 91.0] | 92.8 | 93.4 | 98.6 | 90.4 | 2 | complete 32, pruned 18 | 1.3 |
| wl | hist_gradient_boosting | 98.8 ± 0.5 | 98.8 [98.2, 99.3] | 98.7 | 99.4 | 99.8 | 99.2 | 2 | complete 30, pruned 20 | 51.1 |
| wl | random_forest | 98.6 ± 0.7 | 98.6 [97.8, 99.2] | 98.3 | 99.3 | 99.8 | 99.0 | 2 | complete 43, pruned 7 | 2.6 |
| wl | regresion_logistica | 98.8 ± 0.6 | 98.8 [98.1, 99.4] | 98.7 | 99.4 | 99.8 | 99.1 | 2 | complete 18, pruned 32 | 4.0 |

## Finalistas (mejor clasificador de cada peldaño, F1 macro de algoritmo)

- **e7**: `random_forest`, F1 macro CV 57.3 %, OOF 57.5 % [55.7, 58.9]. Parámetros: `{"n_estimators": 200, "max_depth": null, "max_features": 1.0, "min_samples_leaf": 1}`
- **f3**: `random_forest`, F1 macro CV 84.8 %, OOF 84.8 % [83.4, 85.9]. Parámetros: `{"n_estimators": 300, "max_depth": 16, "max_features": 0.5, "min_samples_leaf": 1}`
- **wl**: `hist_gradient_boosting`, F1 macro CV 97.7 %, OOF 97.8 % [97.1, 98.3]. Parámetros: `{"min_samples_leaf": 50, "max_leaf_nodes": 32, "h": 3, "columnas_wl": 2048, "learning_rate": 0.09703686283829623, "max_iter": 150, "l2_regularization": 0.010690894168621169, "min_circuitos": 5}`

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
| wl#594428 | 2.88 | 0.23 |
| wl#794844 | 2.5 | 0.87 |
| wl#505292 | 1.07 | 0.14 |
| wl#322033 | 0.82 | 0.43 |
| cx | 0.75 | 0.09 |
| wl#706335 | 0.43 | 0.5 |
| wl#302667 | 0.33 | 0.18 |
| wl#321721 | 0.31 | 0.18 |
| wl#462782 | 0.26 | 0.14 |
| wl#195855 | 0.25 | 0.11 |
| wl#856826 | 0.19 | 0.16 |
| wl#173268 | 0.17 | 0.21 |
| wl#98310 | 0.16 | 0.11 |
| wl#411193 | 0.12 | 0.25 |
| wl#294493 | 0.12 | 0.1 |

## Estudios sustituidos por una revisión (rango ampliado)

- `linea_base_v1.e7.hist_gradient_boosting.algoritmo`: F1 macro CV 56.1 % → `linea_base_v1.e7.hist_gradient_boosting.algoritmo.r2` 56.3 %
- `linea_base_v1.e7.regresion_logistica.algoritmo`: F1 macro CV 39.0 % → `linea_base_v1.e7.regresion_logistica.algoritmo.r2` 39.0 %
- `linea_base_v1.f3.hist_gradient_boosting.familia`: F1 macro CV 96.7 % → `linea_base_v1.f3.hist_gradient_boosting.familia.r2` 96.6 %
- `linea_base_v1.f3.regresion_logistica.algoritmo`: F1 macro CV 82.7 % → `linea_base_v1.f3.regresion_logistica.algoritmo.r2` 83.2 %
- `linea_base_v1.f3.regresion_logistica.familia`: F1 macro CV 89.4 % → `linea_base_v1.f3.regresion_logistica.familia.r2` 89.5 %
- `linea_base_v1.wl.hist_gradient_boosting.algoritmo`: F1 macro CV 97.7 % → `linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2` 97.7 %
- `linea_base_v1.wl.hist_gradient_boosting.familia`: F1 macro CV 98.8 % → `linea_base_v1.wl.hist_gradient_boosting.familia.r2` 98.8 %
- `linea_base_v1.wl.random_forest.algoritmo`: F1 macro CV 97.1 % → `linea_base_v1.wl.random_forest.algoritmo.r2` 97.1 %
- `linea_base_v1.wl.random_forest.familia`: F1 macro CV 98.6 % → `linea_base_v1.wl.random_forest.familia.r2` 98.6 %
- `linea_base_v1.wl.regresion_logistica.algoritmo`: F1 macro CV 96.9 % → `linea_base_v1.wl.regresion_logistica.algoritmo.r2` 97.0 %
- `linea_base_v1.wl.regresion_logistica.familia`: F1 macro CV 98.8 % → `linea_base_v1.wl.regresion_logistica.familia.r2` 98.8 %

Tiempo total de búsqueda: 10.1 h.

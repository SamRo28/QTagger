# Diagnósticos de la línea base (fase 10)

Dataset dataset-v1.1 (0f5f28fcf120), commit `75be38c`, equipo `m3pro`. Solo validación cruzada de 5 bloques sobre train + val de `aleatorio`; objetivo algoritmo. Intervalos del 95 % por bootstrap de grupos. **Sin test.**

## 1. Etiquetas barajadas

`linea_base_v1.wl.hist_gradient_boosting.algoritmo` con su mejor configuración (`{"h": 2, "learning_rate": 0.09999227353112966, "l2_regularization": 3.337201737416827e-05, "min_circuitos": 5, "min_samples_leaf": 33, "columnas_wl": 1024, "max_leaf_nodes": 24, "max_iter": 250}`), etiquetas de algoritmo permutadas entre los grupos de la parte de entrenamiento de cada bloque (el bloque retenido conserva las verdaderas). Azar: 5.0 %.

| Semilla | F1 macro por bloque (%) | Media de bloques (%) | OOF (%) |
|---|---|---|---|
| 1 | 3.9, 5.9, 6.0, 3.9, 4.2 | 4.8 | 4.7 |
| 2 | 3.0, 2.8, 4.3, 4.0, 4.0 | 3.6 | 3.6 |
| 3 | 4.6, 6.4, 3.8, 2.5, 8.4 | 5.1 | 5.1 |

## 2. Gemelos estructurales

Un circuito retenido tiene gemelo si algún circuito de la parte de entrenamiento de su bloque tiene la misma huella WL (multiconjunto de etiquetas WL de 64 bits en la iteración h, mismas etiquetas iniciales que el peldaño WL).

### h2: 22.6 % con gemelo; 0.4 % con algún gemelo de otra clase

| Peldaño | Con gemelo: n | F1 macro con gemelo | Sin gemelo: n | F1 macro sin gemelo | Con gemelo, clases comunes | Sin gemelo, clases comunes |
|---|---|---|---|---|---|---|
| e7 | 1828 | 76.7 [73.8, 86.5] | 6243 | 54.2 [52.2, 55.8] | 82.6 [79.7, 84.9] (n 1803) | 72.4 [70.0, 74.5] (n 3557) |
| f3 | 1828 | 82.3 [79.9, 91.6] | 6243 | 83.8 [82.3, 85.0] | 88.6 [86.9, 90.2] (n 1803) | 85.7 [84.0, 87.1] (n 3557) |
| wl | 1828 | 90.2 [89.1, 99.6] | 6243 | 97.6 [96.7, 98.3] | 99.1 [98.5, 99.6] (n 1803) | 98.5 [97.1, 99.4] (n 3557) |

Clases comunes (al menos 20 circuitos con gemelo y otros tantos sin gemelo): bv, cdkm_ripple_carry_adder, dj, qnn, simon, vqe_real_amp, vqe_su2, vqe_two_local. La F1 macro de todas las clases oscila cuando una clase con muy pocos circuitos entra o sale del remuestreo; por eso su intervalo puede quedar casi por encima del valor.

F1 por clase (h2; «con» / «sin» gemelo, %):

| Clase | n | % con gemelo | % gemelo otra clase | e7 con | e7 sin | f3 con | f3 sin | wl con | wl sin |
|---|---|---|---|---|---|---|---|---|---|
| ae | 150 | 0.0 | 0.0 | — | 45.4 [35, 54] | — | 79.1 [72, 84] | — | 100.0 [100, 100] |
| bmw_quark_cardinality | 264 | 0.8 | 0.8 | 0.0 [0, 0] | 56.5 [46, 65] | 0.0 [0, 0] | 95.0 [91, 98] | 0.0 [0, 0] | 100.0 [100, 100] |
| bv | 665 | 72.2 | 1.7 | 89.8 [87, 92] | 45.5 [39, 52] | 93.8 [91, 96] | 72.7 [67, 78] | 99.4 [99, 100] | 98.4 [97, 99] |
| cdkm_ripple_carry_adder | 146 | 64.4 | 0.0 | 100.0 [100, 100] | 69.0 [51, 81] | 100.0 [100, 100] | 92.2 [82, 99] | 100.0 [100, 100] | 93.1 [81, 99] |
| dj | 675 | 30.1 | 0.7 | 79.8 [74, 85] | 49.1 [42, 55] | 85.8 [80, 90] | 83.8 [80, 87] | 98.5 [96, 100] | 98.5 [98, 99] |
| ghz | 149 | 3.4 | 0.0 | 83.3 [0, 100] | 82.1 [75, 88] | 100.0 [100, 100] | 97.9 [96, 100] | 100.0 [100, 100] | 99.7 [99, 100] |
| graphstate | 128 | 0.0 | 0.0 | — | 36.7 [26, 47] | — | 95.7 [92, 98] | — | 98.8 [97, 100] |
| grover | 393 | 0.0 | 0.0 | — | 96.2 [94, 98] | — | 96.9 [95, 99] | — | 98.6 [97, 100] |
| qaoa | 526 | 3.4 | 0.0 | 100.0 [100, 100] | 57.0 [51, 63] | 97.3 [86, 100] | 93.5 [91, 96] | 100.0 [100, 100] | 99.4 [99, 100] |
| qft | 194 | 0.0 | 0.0 | — | 2.7 [1, 5] | — | 80.9 [73, 87] | — | 96.2 [92, 99] |
| qftentangled | 200 | 0.0 | 0.0 | — | 6.5 [3, 10] | — | 85.0 [78, 90] | — | 96.7 [92, 100] |
| qnn | 1048 | 35.9 | 0.0 | 100.0 [100, 100] | 91.2 [89, 93] | 100.0 [100, 100] | 99.9 [100, 100] | 100.0 [100, 100] | 100.0 [100, 100] |
| qpeexact | 171 | 0.0 | 0.0 | — | 14.8 [6, 24] | — | 39.5 [29, 49] | — | 90.8 [86, 94] |
| qpeinexact | 173 | 0.0 | 0.0 | — | 10.7 [4, 18] | — | 44.6 [33, 54] | — | 91.4 [87, 94] |
| simon | 496 | 6.7 | 0.0 | 75.9 [57, 88] | 91.2 [89, 93] | 100.0 [100, 100] | 98.5 [97, 100] | 100.0 [100, 100] | 98.9 [98, 100] |
| superdense_coding | 198 | 0.0 | 0.0 | — | 80.2 [72, 86] | — | 95.5 [91, 98] | — | 97.0 [94, 99] |
| vqe_real_amp | 655 | 16.9 | 0.5 | 33.5 [24, 43] | 50.4 [46, 55] | 44.1 [34, 53] | 62.3 [59, 65] | 97.7 [96, 100] | 98.1 [97, 99] |
| vqe_su2 | 1048 | 37.2 | 0.6 | 83.9 [80, 87] | 64.0 [59, 69] | 86.8 [83, 90] | 73.9 [70, 77] | 99.0 [98, 100] | 100.0 [100, 100] |
| vqe_two_local | 627 | 18.5 | 0.5 | 97.9 [96, 100] | 82.6 [79, 86] | 97.9 [96, 100] | 96.7 [95, 98] | 97.9 [96, 100] | 98.1 [96, 99] |
| wstate | 165 | 0.0 | 0.0 | — | 52.5 [41, 62] | — | 91.6 [86, 96] | — | 98.2 [96, 100] |

### h3: 22.3 % con gemelo; 0.4 % con algún gemelo de otra clase

| Peldaño | Con gemelo: n | F1 macro con gemelo | Sin gemelo: n | F1 macro sin gemelo | Con gemelo, clases comunes | Sin gemelo, clases comunes |
|---|---|---|---|---|---|---|
| e7 | 1799 | 76.8 [73.7, 86.5] | 6272 | 54.2 [52.2, 55.8] | 82.7 [79.7, 84.9] (n 1774) | 72.4 [69.9, 74.5] (n 3586) |
| f3 | 1799 | 82.4 [80.1, 91.7] | 6272 | 83.7 [82.2, 84.9] | 88.7 [87.0, 90.3] (n 1774) | 85.6 [83.9, 87.0] (n 3586) |
| wl | 1799 | 90.2 [89.1, 99.6] | 6272 | 97.6 [96.7, 98.3] | 99.1 [98.4, 99.6] (n 1774) | 98.5 [97.1, 99.4] (n 3586) |

Clases comunes (al menos 20 circuitos con gemelo y otros tantos sin gemelo): bv, cdkm_ripple_carry_adder, dj, qnn, simon, vqe_real_amp, vqe_su2, vqe_two_local. La F1 macro de todas las clases oscila cuando una clase con muy pocos circuitos entra o sale del remuestreo; por eso su intervalo puede quedar casi por encima del valor.

F1 por clase (h3; «con» / «sin» gemelo, %):

| Clase | n | % con gemelo | % gemelo otra clase | e7 con | e7 sin | f3 con | f3 sin | wl con | wl sin |
|---|---|---|---|---|---|---|---|---|---|
| ae | 150 | 0.0 | 0.0 | — | 45.4 [35, 54] | — | 79.1 [72, 84] | — | 100.0 [100, 100] |
| bmw_quark_cardinality | 264 | 0.8 | 0.8 | 0.0 [0, 0] | 56.5 [46, 65] | 0.0 [0, 0] | 95.0 [91, 98] | 0.0 [0, 0] | 100.0 [100, 100] |
| bv | 665 | 72.2 | 1.7 | 90.0 [87, 93] | 45.3 [39, 52] | 94.1 [92, 96] | 72.2 [66, 78] | 99.4 [99, 100] | 98.4 [97, 99] |
| cdkm_ripple_carry_adder | 146 | 64.4 | 0.0 | 100.0 [100, 100] | 69.0 [51, 81] | 100.0 [100, 100] | 92.2 [82, 99] | 100.0 [100, 100] | 93.1 [82, 99] |
| dj | 675 | 29.6 | 0.7 | 80.1 [74, 85] | 49.1 [42, 55] | 86.4 [81, 91] | 83.5 [80, 87] | 98.5 [96, 100] | 98.5 [98, 99] |
| ghz | 149 | 3.4 | 0.0 | 83.3 [0, 100] | 82.1 [75, 88] | 100.0 [100, 100] | 97.9 [96, 100] | 100.0 [100, 100] | 99.7 [99, 100] |
| graphstate | 128 | 0.0 | 0.0 | — | 36.7 [25, 47] | — | 95.7 [92, 98] | — | 98.8 [97, 100] |
| grover | 393 | 0.0 | 0.0 | — | 96.2 [94, 98] | — | 96.9 [95, 99] | — | 98.6 [97, 100] |
| qaoa | 526 | 3.4 | 0.0 | 100.0 [100, 100] | 57.0 [51, 63] | 97.3 [86, 100] | 93.5 [91, 96] | 100.0 [100, 100] | 99.4 [99, 100] |
| qft | 194 | 0.0 | 0.0 | — | 2.7 [1, 5] | — | 80.9 [73, 86] | — | 96.2 [92, 99] |
| qftentangled | 200 | 0.0 | 0.0 | — | 6.5 [3, 10] | — | 85.0 [78, 90] | — | 96.7 [92, 100] |
| qnn | 1048 | 33.4 | 0.0 | 100.0 [100, 100] | 91.5 [89, 93] | 100.0 [100, 100] | 99.9 [100, 100] | 100.0 [100, 100] | 100.0 [100, 100] |
| qpeexact | 171 | 0.0 | 0.0 | — | 14.8 [6, 24] | — | 39.5 [28, 49] | — | 90.8 [86, 94] |
| qpeinexact | 173 | 0.0 | 0.0 | — | 10.7 [4, 17] | — | 44.6 [33, 54] | — | 91.4 [87, 94] |
| simon | 496 | 6.7 | 0.0 | 75.9 [57, 88] | 91.2 [89, 94] | 100.0 [100, 100] | 98.5 [97, 99] | 100.0 [100, 100] | 98.9 [98, 100] |
| superdense_coding | 198 | 0.0 | 0.0 | — | 80.2 [72, 86] | — | 95.5 [91, 98] | — | 97.0 [94, 99] |
| vqe_real_amp | 655 | 16.9 | 0.5 | 33.5 [24, 43] | 50.4 [46, 54] | 44.1 [35, 53] | 62.3 [59, 65] | 97.7 [96, 100] | 98.1 [97, 99] |
| vqe_su2 | 1048 | 37.2 | 0.6 | 83.9 [80, 87] | 64.0 [59, 69] | 86.8 [83, 90] | 73.9 [70, 77] | 99.0 [98, 100] | 100.0 [100, 100] |
| vqe_two_local | 627 | 18.5 | 0.5 | 97.9 [96, 100] | 82.6 [79, 86] | 97.9 [96, 100] | 96.7 [95, 98] | 97.9 [96, 100] | 98.1 [96, 99] |
| wstate | 165 | 0.0 | 0.0 | — | 52.5 [41, 62] | — | 91.6 [86, 96] | — | 98.2 [96, 100] |

## 3. Confusión de `linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2` (122 errores fuera de bloque)

| Clase A | Clase B | Total | A → B | B → A | Soporte A | Soporte B |
|---|---|---|---|---|---|---|
| qpeexact | qpeinexact | 25 | 14 | 11 | 171 | 173 |
| vqe_real_amp | vqe_two_local | 25 | 16 | 9 | 655 | 627 |
| bmw_quark_cardinality | vqe_su2 | 8 | 2 | 6 | 264 | 1048 |
| bv | dj | 7 | 1 | 6 | 665 | 675 |
| simon | superdense_coding | 7 | 3 | 4 | 496 | 198 |

Matriz completa (filas: clase real; columnas: predicha, en el mismo orden):

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 ae | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 bmw_quark_cardinality | 0 | 262 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| 2 bv | 0 | 0 | 660 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| 3 cdkm_ripple_carry_adder | 0 | 0 | 0 | 141 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |
| 4 dj | 0 | 0 | 6 | 0 | 667 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 5 ghz | 0 | 0 | 0 | 0 | 0 | 149 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 graphstate | 0 | 0 | 0 | 0 | 0 | 0 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 grover | 0 | 0 | 0 | 1 | 3 | 1 | 0 | 384 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 qaoa | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 525 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 9 qft | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 2 | 188 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 10 qftentangled | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 5 | 192 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 11 qnn | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1048 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 12 qpeexact | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 153 | 14 | 0 | 0 | 0 | 0 | 0 | 1 |
| 13 qpeinexact | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 11 | 159 | 0 | 0 | 0 | 0 | 0 | 0 |
| 14 simon | 0 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 490 | 3 | 0 | 0 | 0 | 0 |
| 15 superdense_coding | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 4 | 192 | 0 | 0 | 0 | 0 |
| 16 vqe_real_amp | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 639 | 0 | 16 | 0 |
| 17 vqe_su2 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1042 | 0 | 0 |
| 18 vqe_two_local | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 | 618 | 0 |
| 19 wstate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 162 |

### F1 macro por nivel de los finalistas (%)

| Peldaño | Estudio | alg | indep | nativegates | mapped |
|---|---|---|---|---|---|
| e7 | `linea_base_v1.e7.random_forest.algoritmo` | 56.8 [54.9, 58.5] (n 4955) | 48.3 [39.9, 54.7] (n 1106) | 51.1 [46.5, 54.8] (n 1019) | 45.2 [41.0, 48.4] (n 991) |
| f3 | `linea_base_v1.f3.random_forest.algoritmo` | 86.1 [84.5, 87.4] (n 4955) | 71.7 [67.2, 81.2] (n 1106) | 87.2 [81.7, 90.5] (n 1019) | 60.8 [57.0, 64.1] (n 991) |
| wl | `linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2` | 99.0 [98.4, 99.4] (n 4955) | 82.8 [80.8, 98.9] (n 1106) | 95.4 [90.5, 98.3] (n 1019) | 86.0 [79.7, 90.4] (n 991) |

Solo clases con al menos 20 circuitos en los cuatro niveles (bmw_quark_cardinality, qft, qftentangled, qnn, qpeexact, qpeinexact, superdense_coding, vqe_su2):

| Peldaño | alg | indep | nativegates | mapped |
|---|---|---|---|---|
| e7 | 46.2 [43.4, 48.8] (n 1314) | 48.4 [44.1, 52.7] (n 1062) | 42.7 [39.3, 45.9] (n 460) | 49.9 [45.4, 54.1] (n 460) |
| f3 | 83.5 [80.3, 86.5] (n 1314) | 79.6 [75.3, 83.7] (n 1062) | 84.7 [80.9, 88.1] (n 460) | 66.0 [61.6, 69.9] (n 460) |
| wl | 98.7 [97.5, 99.6] (n 1314) | 99.1 [98.2, 99.7] (n 1062) | 98.3 [96.7, 99.6] (n 460) | 88.3 [84.8, 91.4] (n 460) |

## 4. Rangos de hiperparámetros

| Estudio | F1 macro CV (%) | Valores en el extremo |
|---|---|---|
| `linea_base_v1.e7.hist_gradient_boosting.algoritmo` | 56.1 | min_samples_leaf = 7 (inferior, 0.04); max_leaf_nodes = 62 (superior, 0.98) |
| `linea_base_v1.e7.hist_gradient_boosting.algoritmo.r2` | 56.3 | max_leaf_nodes = 123 (superior, 0.96) |
| `linea_base_v1.e7.hist_gradient_boosting.familia` | 74.8 | — |
| `linea_base_v1.e7.random_forest.algoritmo` | 57.3 | min_samples_leaf = 1 (inferior, 0.00) — límite natural |
| `linea_base_v1.e7.random_forest.familia` | 75.7 | min_samples_leaf = 1 (inferior, 0.00) — límite natural |
| `linea_base_v1.e7.regresion_logistica.algoritmo` | 39.0 | C = 332.2 (superior, 0.92) |
| `linea_base_v1.e7.regresion_logistica.algoritmo.r2` | 39.0 | — |
| `linea_base_v1.e7.regresion_logistica.familia` | 42.2 | — |
| `linea_base_v1.f3.hist_gradient_boosting.algoritmo` | 83.9 | — |
| `linea_base_v1.f3.hist_gradient_boosting.familia` | 96.7 | learning_rate = 0.2844 (superior, 0.98); l2_regularization = 3.381e-06 (inferior, 0.09); min_samples_leaf = 47 (superior, 0.93); max_iter = 300 (superior, 1.00) |
| `linea_base_v1.f3.hist_gradient_boosting.familia.r2` | 96.6 | min_samples_leaf = 139 (superior, 0.92) |
| `linea_base_v1.f3.random_forest.algoritmo` | 84.8 | min_samples_leaf = 1 (inferior, 0.00) — límite natural |
| `linea_base_v1.f3.random_forest.familia` | 96.1 | — |
| `linea_base_v1.f3.regresion_logistica.algoritmo` | 82.7 | C = 903.7 (superior, 0.99) |
| `linea_base_v1.f3.regresion_logistica.algoritmo.r2` | 83.2 | C = 1.769e+04 (superior, 0.91) |
| `linea_base_v1.f3.regresion_logistica.familia` | 89.4 | C = 640.8 (superior, 0.97) |
| `linea_base_v1.f3.regresion_logistica.familia.r2` | 89.5 | — |
| `linea_base_v1.wl.hist_gradient_boosting.algoritmo` | 97.7 | columnas_wl = 1024 (superior, [256, 512, 1024]) |
| `linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2` | 97.7 | min_samples_leaf = 50 (superior, 1.00); h = 3 (superior, 1.00) |
| `linea_base_v1.wl.hist_gradient_boosting.familia` | 98.8 | learning_rate = 0.2763 (superior, 0.97); min_circuitos = 2 (inferior, [2, 5, 10]); columnas_wl = 1024 (superior, [256, 512, 1024]); max_leaf_nodes = 19 (inferior, 0.08); max_iter = 50 (inferior, 0.00) |
| `linea_base_v1.wl.hist_gradient_boosting.familia.r2` | 98.8 | min_samples_leaf = 49 (superior, 0.98); max_leaf_nodes = 59 (superior, 0.93); h = 1 (inferior, 0.00); min_circuitos = 1 (inferior, [1, 2, 5, 10]) |
| `linea_base_v1.wl.random_forest.algoritmo` | 97.1 | min_samples_leaf = 1 (inferior, 0.00) — límite natural; min_circuitos = 10 (superior, [2, 5, 10]) |
| `linea_base_v1.wl.random_forest.algoritmo.r2` | 97.1 | min_samples_leaf = 1 (inferior, 0.00) — límite natural |
| `linea_base_v1.wl.random_forest.familia` | 98.6 | h = 3 (superior, 1.00); min_samples_leaf = 1 (inferior, 0.00) — límite natural; min_circuitos = 10 (superior, [2, 5, 10]) |
| `linea_base_v1.wl.random_forest.familia.r2` | 98.6 | min_samples_leaf = 1 (inferior, 0.00) — límite natural; n_estimators = 100 (inferior, 0.00) |
| `linea_base_v1.wl.regresion_logistica.algoritmo` | 96.9 | min_circuitos = 10 (superior, [2, 5, 10]) |
| `linea_base_v1.wl.regresion_logistica.algoritmo.r2` | 97.0 | — |
| `linea_base_v1.wl.regresion_logistica.familia` | 98.8 | C = 0.003418 (inferior, 0.09) |
| `linea_base_v1.wl.regresion_logistica.familia.r2` | 98.8 | — |

Finalistas originales: e7 `linea_base_v1.e7.random_forest.algoritmo` (57.3 %), f3 `linea_base_v1.f3.random_forest.algoritmo` (84.8 %), wl `linea_base_v1.wl.hist_gradient_boosting.algoritmo` (97.7 %)

Finalistas actuales: e7 `linea_base_v1.e7.random_forest.algoritmo` (57.3 %), f3 `linea_base_v1.f3.random_forest.algoritmo` (84.8 %), wl `linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2` (97.7 %)

**Cambia algún finalista: sí** (linea_base_v1.wl.hist_gradient_boosting.algoritmo, linea_base_v1.wl.hist_gradient_boosting.algoritmo.r2)

# Plan de construcción: fases de modelos (9 a 18)

> **Instrucciones para la IA que ejecute este plan.**
>
> - Lee el documento entero y `docs/DECISIONES.md` antes de escribir código.
> - Trabaja **fase a fase**. Al terminar cada fase, ejecuta sus comprobaciones y sus tests, enséñame un resumen y **espera mi confirmación** antes de pasar a la siguiente.
> - No inventes APIs: comprueba la versión instalada de cada librería y adapta el código. Si algo del plan no encaja con la API real o puede mejorarse claramente, **detente y pregúntame**.
> - Regla absoluta: **nada se evalúa en test ni en los desconocidos hasta la fase 17**, y solo mediante `qcclass evaluar-final`. Cualquier análisis posterior a la fase 17 se guarda aparte y se etiqueta como «análisis posterior».

---

## 0. Contexto y decisiones de diseño

La etapa de datos (fases 1 a 8) está terminada y la medición de rendimiento está hecha. Las decisiones de diseño de los modelos se tomaron paso a paso y están razonadas en el documento del proyecto. Aquí se resumen las que condicionan el código.

| Tema | Decisión |
|---|---|
| Versionado | Etiqueta git por versión del dataset y `data/VERSION.json` con huellas. Esta etapa crea `dataset-v1.1` |
| Test | Protegido con `--evaluar-test` y registro en `results/accesos_test.log`. Esto se amplía al conjunto `desconocido` |
| Registro de experimentos | MLflow con servidor en el M1 Pro, más un resumen Markdown versionado en git |
| Unidad de evaluación | Por circuito, con intervalos por bootstrap de grupos (se remuestrean grupos completos) |
| Métrica principal | **F1 macro de algoritmo**. Secundarias: exactitud balanceada, exactitud, top-3, MCC y matriz de confusión. Las mismas métricas para la familia |
| Selección de hiperparámetros | Validación cruzada por grupos, 5 bloques, sobre train + val del protocolo `aleatorio`, con 1 semilla por bloque. Los hiperparámetros elegidos se reutilizan en todos los protocolos |
| Modelos finales | 10 semillas, entrenados con train + val y número de épocas fijo. **Excepción: el protocolo `abierto` se entrena solo con train**, porque su val se usa para fijar el umbral de desconocidos y poner a punto la explicabilidad |
| Etiquetas | −1 = solo familia (la cabeza de algoritmo lo ignora). −2 = desconocido (nunca se entrena). Las variantes `solo_evaluacion` nunca entran en train ni en las métricas principales |
| Línea base | Tres peldaños: solo tamaño (E7), F3 y WL. Tres clasificadores en cada uno: regresión logística, Random Forest y HistGradientBoosting. Optuna con 50 intentos |
| GNN | GINE o GATv2; 64 o 128 dimensiones; de 3 a 8 capas; residuales, normalización y Jumping Knowledge fijos; nodo virtual opcional; aristas en ambos sentidos con la dirección marcada; agregación media + máximo o atención (sin suma); dos cabezas con L = L_alg + λ · L_fam |
| Entrenamiento | AdamW; calendario en coseno con calentamiento; dropout y regularización de pesos; sin suavizado de etiquetas; épocas fijas (50, 100 o 150) elegidas por Optuna; lotes de 100 000 nodos; precisión mixta (sujeta a comprobación); 4 procesos de carga; puntos de guardado en cada época |
| Búsqueda | 50 intentos, TPE, recorte por mediana tras 10 intentos completos, base de datos de Optuna compartida en el M1 Pro |
| Ablaciones | Seis ablaciones medidas sin tocar el test. Red convolucional 1D sobre F6 con prioridad baja |
| Explicabilidad | Gradientes integrados promediados sobre 10 semillas, contrastados con GNNExplainer y oclusión de bloques. Métricas: AUROC y precisión en los k primeros, frente a importancia aleatoria y la regla «todas las CNOT». Motivos frecuentes por huella WL. Definiciones fijadas en el pre-registro |
| Desconocidos | Mahalanobis como método principal; probabilidad máxima y energía como secundarios. Umbral que acepta el 95 % de los conocidos de val. Familia de `hhl` y `hrs`. Agrupación con HDBSCAN y medida de pureza |
| Cierre | Pre-registro en `docs/PLAN_ANALISIS.md` con etiqueta git; en test solo los finalistas; bootstrap pareado por grupos (10 000 remuestreos), tres comparaciones principales y corrección de Holm |

**Equipos:**

- **RTX 3050 de portátil, 4 GB, en WSL2** (`rtx3050`): equipo principal de la GNN. Una época de la red de referencia tarda 3,7 s con precisión mixta y 4 procesos de carga. WSL2 tiene 11 GB de RAM (configurado en `.wslconfig`).
- **MacBook Pro M3 Pro, 18 GB** (`m3pro`): línea base, WL, análisis y, si hace falta, entrenamiento de la GNN **en CPU**. MPS queda descartado porque resultó más lento que la CPU.
- **MacBook Pro M1 Pro, 16 GB** (`m1pro`): servidor de MLflow y de la base de datos de Optuna. Debe estar encendido y sin entrar en reposo mientras haya experimentos en marcha.

---

## Fase 9. Infraestructura común

### Tareas

1. **`dataset-v1.1`**
   - Crea `data/splits/aleatorio_bloques.csv` (`group_id`, `bloque` de 0 a 4) con `StratifiedGroupKFold` sobre train + val del protocolo `aleatorio`, estratificado por etiqueta y con semilla fija. Las variantes `solo_evaluacion` heredan el bloque de su grupo, pero nunca se entrenan.
   - Amplía `data/VERSION.json` con la huella del archivo de bloques y con las versiones de Python, Qiskit, MQT Bench, torch, PyTorch Geometric, NumPy, scikit-learn y Optuna.
   - Haz que `qcclass verificar` lo compruebe todo y crea la etiqueta git `dataset-v1.1`.
2. **Servidores en el M1 Pro**
   - Un servidor de MLflow accesible desde la red local, con su base de datos y los artefactos en el disco del M1 Pro, servidos a través del propio servidor.
   - Una base de datos de Optuna compartida, accesible desde los otros dos equipos. Elige la opción que soporte la versión instalada de Optuna (una base de datos relacional en servidor, o el proxy de almacenamiento de Optuna si existe en esa versión) y justifícalo en `docs/DECISIONES.md`.
   - Comandos para arrancar y parar los servidores (`make servidores`, `make parar-servidores`) e instrucciones para evitar el reposo del M1 Pro.
   - Los clientes se configuran con variables de entorno en un `.env` no versionado: `QCCLASS_MLFLOW_URI` y `QCCLASS_OPTUNA_STORAGE`. Si no se puede conectar, el programa aborta con un mensaje claro; nunca registra en local en silencio.
   - Un comando `qcclass ping` que compruebe la conexión con ambos servidores.
3. **Módulo de evaluación común** (`src/qcclass/evaluacion/`)
   - Métricas: F1 macro (principal), exactitud balanceada, exactitud, exactitud top-3, MCC y matriz de confusión, para algoritmo y para familia.
   - Reglas de exclusión: −2 nunca cuenta en las métricas de conocidos; −1 solo cuenta en familia; `solo_evaluacion` se excluye de las métricas principales.
   - Desgloses por nivel, por tamaño, por `construccion_mcx` (grover), por `tipo_oraculo_dj` (dj) y por familia.
   - Agregación de predicciones fuera de bloque (validación cruzada).
   - Bootstrap por grupos (10 000 remuestreos) para intervalos del 95 %, bootstrap **pareado** por grupos para diferencias entre dos modelos, y corrección de Holm.
   - Salida en JSON y Markdown.
   - Tests con datos sintéticos, que incluyan los dos ejemplos didácticos del diseño:
     - 300 DJ y 30 QFT, con la mitad de los QFT confundidos con DJ: exactitud 95,5 %, exactitud balanceada 75 % y F1 macro 82,1 %.
     - 300 DJ, 30 QFT y 30 GHZ, con 60 DJ llamados QFT: exactitud balanceada 93,3 % y F1 macro 79,6 %.
4. **Protección del test.** Amplía `--evaluar-test` y el registro de accesos para que cubran también el conjunto `desconocido` de `abierto.csv`. Solo `qcclass evaluar-final` (fase 17) puede usarlo.
5. **Utilidades de entrenamiento.**
   - Semillas globales.
   - Puntos de guardado en cada época y reanudación tras un corte.
   - Registro de la información del equipo, reutilizando lo de `qcclass medir`.
   - Registro en MLflow, en cada ejecución, de la versión del dataset, el commit de git, el equipo y la configuración completa.
6. Renombra en `results/medicion/` el equipo `rtx3035` a `rtx3050`.

### Comprobación

- Tests en verde; `qcclass verificar` correcto en los tres equipos.
- `qcclass ping` funciona desde la RTX 3050 y desde el M3 Pro.
- Una ejecución de prueba aparece en la interfaz de MLflow.
- Un estudio de Optuna de prueba recibe 2 intentos desde la RTX 3050 y 2 desde el M3 Pro, y los cuatro aparecen en el mismo estudio.

---

## Fase 10. Línea base (en el M3 Pro)

### Tareas

1. **Peldaños.**
   - **Solo tamaño (E7):** `n_qubits`, `profundidad`, `n_puertas` de F3.
   - **F3 completo.**
   - **WL:**
     - Etiqueta inicial de cada nodo: tipo de puerta, más el ángulo discretizado en las `rz`. Por ejemplo, múltiplos de π/16 y una categoría «otro»; justifica la elección.
     - El rol de control o destino entra como etiqueta de la arista.
     - Número de iteraciones h ∈ {1, 2, 3} como hiperparámetro.
     - Vector de frecuencias normalizado por el número de nodos, con *hashing* a dimensión fija (por ejemplo, 2^14) para no construir un vocabulario que mire datos de validación.
2. **Clasificadores.**
   - Regresión logística multinomial con estandarización.
   - Random Forest.
   - HistGradientBoosting.
   - Pesos de clase calculados sobre la parte de entrenamiento de cada bloque, con la misma fórmula y el mismo tope (3 × mediana) que `balanceo_aleatorio.json`.
3. **Objetivos.** Algoritmo (`y_algoritmo` ≥ 0) y familia (`y_familia` ≥ 0, incluidas las filas con −1). Una búsqueda por peldaño, clasificador y objetivo: 18 estudios.
4. **Optuna.** 50 intentos por estudio, TPE con semilla y objetivo = F1 macro medio de la validación cruzada de 5 bloques. Define espacios de búsqueda razonables y documéntalos.
5. **Salidas.**
   - Predicciones fuera de bloque del mejor intento de cada estudio.
   - Tabla resumen.
   - **Finalistas:** el mejor clasificador de cada peldaño según el F1 macro de algoritmo en validación cruzada.
   - Importancia por permutación de los finalistas.
   - Todo en MLflow y en `results/baseline/RESUMEN.md`.

### Comprobación

Test de integración con `data_mini`; resultado de E7 visible en el resumen. **Sin test.**

---

## Fase 11. Implementación de la GNN

### Tareas

1. **Transformaciones de aristas.**
   - Por defecto: ambos sentidos, con un atributo de dirección (adelante o atrás), además del rol de control o destino.
   - Variantes para las ablaciones: `solo_adelante`, `sin_direccion`, `sin_rol` y `sin_angulos` (seno y coseno a cero).
2. **Modelo** (`src/qcclass/modelos/gnn.py`), con configuración en un *dataclass*:
   - `tipo_capa` {gine, gatv2}, ambas leyendo `edge_attr`; `dim` {64, 128}; `capas` de 3 a 8; `dropout`.
   - `nodo_virtual` (sí/no); `agregacion` {media_max, atencion}; `jumping_knowledge` (sí por defecto, desactivable para la ablación).
   - Residuales y normalización en todas las capas.
   - Dos cabezas: algoritmo (20 clases) y familia (6). `lambda_familia` (0 para la ablación).
   - **Trazas:** si se le pide, el `forward` devuelve además un diccionario con los vectores de cada nodo en cada capa, el estado del nodo virtual, la salida de Jumping Knowledge, el vector resumen del circuito, los logits y probabilidades de ambas cabezas y los pesos de atención (de GATv2 y de la agregación por atención, si existen). Lo usan la explicabilidad, los desconocidos y el proyecto visual futuro.
   - Un método que describa la arquitectura: capas, dimensiones y parámetros por capa.
3. **Pérdida.** Entropía cruzada de algoritmo con `ignore_index=-1` más λ · entropía cruzada de familia, ambas con pesos de clase calculados sobre los datos de entrenamiento de cada ejecución. Comprueba que ningún −2 entra en el entrenamiento.
4. **Entrenamiento.**
   - AdamW; calendario en coseno con calentamiento (5 % de las épocas).
   - Sampler por presupuesto de nodos (100 000) y 4 procesos de carga (configurable por equipo).
   - Precisión mixta en CUDA. Dispositivo: CUDA si está disponible y, si no, CPU. MPS desactivado por defecto, con el motivo documentado.
   - Puntos de guardado en cada época y reanudación.
   - Registro por época en MLflow: pérdida, F1 en el bloque o conjunto de evaluación cuando exista, tiempo, memoria de GPU y temperatura.
5. **Comprobación de la precisión mixta.** Entrena la configuración de referencia (GINE, 64 dimensiones, 3 capas, 100 épocas) en el bloque 0, con y sin precisión mixta. Si el F1 macro difiere en más de 1 punto, desactívala por defecto y regístralo en `docs/DECISIONES.md`.
6. **Tests:**
   - Formas de las salidas y de las trazas.
   - `ignore_index` y exclusión de −2.
   - Que el sampler respete el presupuesto.
   - Que reanudar desde un punto de guardado dé el mismo estado que el entrenamiento continuo (con tolerancia).
   - Entrenamiento corto con `data_mini` en CPU.

### Comprobación

Entrena la configuración de referencia en un bloque en la RTX 3050 y compara el tiempo por época con la medición (unos 3,7 s, más la evaluación).

---

## Fase 12. Búsqueda de hiperparámetros de la GNN

### Tareas

1. **Estudio** `gnn_aleatorio_v1` en el almacenamiento compartido. TPE con semilla. Recorte por mediana a partir de 10 intentos completos.
   - Valores intermedios: el F1 macro tras cada bloque y, dentro del primer bloque, también cada 10 épocas.
2. **Espacio de búsqueda:**

   | Hiperparámetro | Valores |
   |---|---|
   | `tipo_capa` | gine, gatv2 |
   | `dim` | 64, 128 |
   | `capas` | 3 a 8 |
   | `nodo_virtual` | sí, no |
   | `agregacion` | media_max, atencion |
   | `lambda_familia` | 0,2 a 1 |
   | Tasa de aprendizaje | 0,0001 a 0,003 (logarítmica) |
   | `dropout` | 0 a 0,5 |
   | Regularización de pesos | 0,000001 a 0,01 (logarítmica) |
   | Épocas | 50, 100, 150 |

   Fijos: residuales y normalización, Jumping Knowledge, aristas en ambos sentidos con dirección, lotes de 100 000 nodos, AdamW, coseno con calentamiento y precisión mixta según la fase 11.
3. **Objetivo:** F1 macro medio de algoritmo en los 5 bloques, con 1 semilla por bloque. Registra también el de familia.
4. **Guarda**, de cada intento completado, los modelos de los 5 bloques (último punto de guardado) y sus predicciones fuera de bloque.
5. **Equipos.** Empieza solo con la RTX 3050. Tras 10 intentos, estima el tiempo restante. Si supera dos días, **avísame** antes de añadir el M3 Pro (CPU) como segundo trabajador.
6. **Salidas.**
   - `configs/experimentos/gnn_mejor.yaml`.
   - Informe con el historial, la importancia de los hiperparámetros y gráficos de coordenadas paralelas, guardados también como datos (CSV) para el proyecto visual.
   - Comparación exploratoria, con las predicciones fuera de bloque, entre la GNN y los finalistas de la línea base.
7. **Punto de control (contingencia).** Si alguna clase tiene un F1 fuera de bloque inferior a 0,5, o la GNN queda más de 2 puntos de F1 macro por debajo de la mejor línea base, **para y avísame**. Decidiremos si activar el plan de contingencia (más variantes de compilación para las clases pequeñas) antes de seguir.

### Comprobación

Informe de la búsqueda con el tiempo real consumido. **Sin test.**

---

## Fase 13. Ablaciones y modelo secuencial

### Tareas

1. **Seis ablaciones** sobre la mejor configuración, **sin volver a ajustar** sus hiperparámetros:
   1. sin ángulos;
   2. aristas solo hacia delante, y aristas sin dirección (dos variantes);
   3. sin rol de control o destino;
   4. F5 (grafo de cúbits) en lugar de F4;
   5. sin cabeza de familia (λ = 0);
   6. sin Jumping Knowledge.

   Cada una se mide (a) con validación cruzada de 5 bloques en `aleatorio` (1 semilla) y (b) en `por_tamano`, entrenando con su train y evaluando en su val, con 5 semillas.
2. **Tabla de ablaciones:** diferencia de F1 macro respecto al modelo completo, con intervalos por bootstrap pareado por grupos. Es un análisis exploratorio.
3. **Red convolucional 1D sobre F6 (prioridad baja).** **Pregúntame antes de empezarla.**
   - Cada token se representa con el tipo de puerta, el seno y coseno del ángulo y el índice de cúbit normalizado.
   - Convoluciones 1D con dilatación, agregación global y las mismas dos cabezas.
   - Optuna con 50 intentos y validación cruzada.

### Comprobación

Tabla de ablaciones en `results/ablaciones/RESUMEN.md`. **Sin test.**

---

## Fase 14. Pre-registro

Redacta `docs/PLAN_ANALISIS.md`, **pero no crees la etiqueta**: lo revisaremos antes. Debe contener:

1. **Finalistas que irán a test:** el mejor clasificador de cada peldaño, la GNN y, si se hizo, la red convolucional 1D.
2. **Tres comparaciones principales**, con hipótesis direccionales:
   - GNN frente al mejor WL en `aleatorio`;
   - GNN frente al mejor WL en `por_tamano`;
   - GNN frente a E7 en `aleatorio`.

   Método: bootstrap pareado por grupos con 10 000 remuestreos, intervalo del 95 % de la diferencia de F1 macro y corrección de Holm sobre las tres. El resto de comparaciones se declara exploratorio.
3. **Métricas y desgloses** de todos los protocolos.
4. **Explicabilidad:**
   - Una tabla con el **bloque característico** de cada algoritmo con F7. Por ejemplo, en Grover el difusor y no el oráculo.
   - Una tabla con los **patrones de Leymann esperados** por algoritmo, basada en Shen et al. (arXiv:2501.15895), excluyendo la medición post-selectiva porque las medidas finales se eliminan.
   - Métricas (AUROC y precisión en los k primeros, con k = tamaño del bloque) y referencias (importancia aleatoria y regla «todas las CNOT»).
   - Umbral para definir las «puertas importantes» en la búsqueda de motivos.
5. **Desconocidos:**
   - Método principal: Mahalanobis con covarianza compartida sobre el vector resumen. Secundarios: probabilidad máxima y energía.
   - Umbral: el que acepta el 95 % de los conocidos de val con los modelos de `abierto`.
   - Parámetros de HDBSCAN fijados de antemano (por ejemplo, `min_cluster_size`).
   - Métricas: AUROC, falsos aceptados al 95 %, F1 de la clase desconocido, desglose por algoritmo, acierto de familia en `hhl` y `hrs_cumulative_multiplier`, y pureza de los grupos.
6. **Regla de cambios posteriores:** cualquier análisis tras la fase 17 se guarda en `results/posterior/` y se etiqueta como análisis posterior.

Cuando lo aprobemos, crea la etiqueta `preregistro-v1`.

---

## Fase 15. Modelos finales

### Tareas

1. **GNN:** la mejor configuración, con 10 semillas y el número de épocas elegido.
   - `aleatorio`, `por_tamano`, `oraculos_no_vistos` y `por_nivel`: entrenamiento con train + val.
   - `abierto`: entrenamiento **solo con train**.
2. **Finalistas de la línea base:** entrenamiento con train + val en `aleatorio`, `por_tamano`, `oraculos_no_vistos` y `por_nivel`, con 10 semillas en los modelos con aleatoriedad.
3. **Guarda cada modelo** con sus pesos, su configuración, el commit de git y la versión del dataset en `results/modelos_finales/`, y regístralo en MLflow.
4. **No evalúes nada en test.** Solo registra métricas de entrenamiento como comprobación de sanidad.

---

## Fase 16. Puesta a punto en validación

Todo en esta fase se hace **sobre val del protocolo `abierto`**, con sus modelos (entrenados solo con train), siguiendo exactamente lo pre-registrado.

### Tareas

1. **Explicabilidad.**
   - Gradientes integrados con Captum (atributos a cero como referencia, 50 pasos), promediados sobre las 10 semillas.
   - GNNExplainer sobre 3 semillas.
   - Oclusión de bloques completos de F7.
   - Métricas frente a los bloques característicos y frente a las dos referencias.
   - Búsqueda de motivos: subgrafos conectados de puertas importantes, huella WL (tipo de puerta y ángulo discretizado), recuento por clase y dibujo en SVG de los 3 motivos más frecuentes de cada clase.
2. **Desconocidos.**
   - Medias por clase y covarianza compartida del vector resumen, calculadas con los datos de train.
   - Puntuaciones de Mahalanobis, probabilidad máxima y energía en los conocidos de val.
   - Umbral al 95 %.
   - **No mires ningún desconocido.**
3. **Salidas** en `results/dev/`.

---

## Fase 17. Evaluación final única

### Tareas

Implementa `qcclass evaluar-final`, que:

1. **Comprueba** que existe la etiqueta `preregistro-v1`, que la versión del dataset coincide y que están todos los modelos finales. Si falta algo, aborta.
2. **Evalúa en test** los finalistas en todos los protocolos, con todas las métricas, desgloses e intervalos.
3. **Calcula las tres comparaciones principales** con bootstrap pareado y corrección de Holm.
4. **En `abierto`:**
   - Detección de desconocidos sobre los 552 circuitos desconocidos, con el umbral de la fase 16.
   - Acierto de familia en `hhl` y `hrs_cumulative_multiplier`.
   - Agrupación con HDBSCAN y pureza.
5. **Explicabilidad sobre las 556 variantes etiquetadas de test**, con los modelos finales de `aleatorio`.
6. **Registra** cada acceso, escribe `results/final/` (JSON, `RESUMEN.md` y figuras) y lo registra todo en MLflow.

Se ejecuta **una sola vez**. Lo que venga después es análisis posterior.

---

## Fase 18. Publicación y artefactos para el proyecto visual

### Tareas

1. **Paquetes de publicación.** Los prepara la IA; la subida la decido yo cuando sepa el formato final.
   - Código: `LICENSE` (MIT), `CITATION.cff` y una sección del README para reproducir todo desde cero. Etiqueta de *release*.
   - Dataset para Zenodo: circuitos canónicos (QPY y QASM), metadatos, particiones, bloques, etiquetas y `VERSION.json` en un archivo comprimido con su ficha de metadatos, incluida la atribución a MQT Bench y su licencia.
   - Modelos finales (pesos y configuraciones) y una exportación del resumen de MLflow en CSV.
2. **`qcclass exportar-demo` y `docs/ESQUEMA_DEMO.md`** (esquema fijo y documentado):
   - **Circuitos de ejemplo:** 2 de test por clase (incluido un error cuando exista) y 1 por cada algoritmo desconocido. Para cada uno:
     - QASM y dibujo en SVG;
     - lista de puertas con índice de nodo en F4, tipo, cúbits, columna en el dibujo, ángulo y bloque F7;
     - trazas completas del modelo, con una proyección PCA a 2D de los vectores de cada capa para animarla;
     - importancias por gradientes integrados (media y desviación sobre semillas) y oclusión por bloque;
     - distancias de Mahalanobis por clase y veredicto de desconocido.
   - **Arquitectura:** descripción por capa y ruta a los pesos.
   - **Vistas globales:** proyección UMAP 2D de los vectores resumen de test y desconocidos con sus etiquetas; curvas de entrenamiento; historial e importancias de Optuna; matrices de confusión; tabla de ablaciones.
3. **`docs/GLOSARIO.md`:** déjalo creado con la lista de conceptos (métricas, validación cruzada, bootstrap, regularización, tasa de aprendizaje, coseno con calentamiento, lotes, precisión mixta, procesos de carga, paso de mensajes, GINE, GATv2, atención, Jumping Knowledge, nodo virtual, sobresuavizado, agregación global, WL, gradientes integrados, GNNExplainer, oclusión, Mahalanobis, energía, HDBSCAN, pureza, pre-registro y corrección de Holm). El contenido lo redactaremos aparte.

---

## Calendario orientativo de cómputo

| Fase | Equipo | Duración estimada |
|---|---|---|
| 9 | — | Implementación |
| 10 | M3 Pro | 3–6 h (18 estudios) |
| 11 | RTX 3050 | Minutos (comprobaciones) |
| 12 | RTX 3050 (+ M3 Pro) | ~25 h |
| 13 | RTX 3050 | ~20 h (+ 5–10 h si se hace la red convolucional 1D) |
| 15 | RTX 3050 y M3 Pro | ~6 h |
| 16 | RTX 3050 | ~3 h |
| 17 | RTX 3050 | ~1 h |

En total, entre dos y tres días de máquina. Los entrenamientos largos se lanzan con los equipos enchufados y ventilados, preferiblemente de noche.

---

## Reglas generales

- Un commit por fase cuando pasen sus comprobaciones, y `docs/DECISIONES.md` actualizado en ese mismo commit.
- Toda ejecución registra en MLflow la versión del dataset, el commit, el equipo y la configuración.
- Nunca se usa `--evaluar-test` fuera de la fase 17.
- Las configuraciones de los experimentos van en `configs/experimentos/`, una por experimento. Nada de rutas ni hiperparámetros escritos a mano en el código.
- Si una decisión de este plan resulta imposible o claramente mejorable al ver la API real o los datos, **detente y pregúntame**.
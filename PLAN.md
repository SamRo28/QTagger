# Plan de construcción del repositorio: clasificación de circuitos cuánticos por algoritmo

> **Instrucciones para la IA que ejecute este plan.** Lee el documento entero antes de escribir código. Trabaja **fase a fase**: al terminar cada fase, ejecuta sus comprobaciones y sus tests, enséñame un resumen de lo que has hecho y **espera mi confirmación** antes de pasar a la siguiente. No inventes APIs: antes de usar una librería, comprueba la versión instalada (`pip show`, `python -c "import x; help(x.y)"`) y adapta el código a esa versión. Si algo de este plan no encaja con la API real, avísame y propón la alternativa en lugar de improvisar.

---

## 0. Contexto del proyecto

Se quiere entrenar una red neuronal de grafos (GNN) que, a partir de un circuito cuántico **aplanado en puertas básicas**, identifique qué algoritmo implementa (Grover, QFT, QAOA, Deutsch-Jozsa…). Este repositorio cubre, en esta primera etapa, **la estructura del proyecto y la generación del dataset** en todos sus formatos. Los modelos llegan en fases posteriores (sección 9).

Fuente principal de circuitos: **MQT Bench** (paquete `mqt.bench`, API 2.x), que ofrece cada algoritmo en cuatro niveles de abstracción: `ALG` (algorítmico), `INDEP` (independiente del hardware), `NATIVEGATES` (puertas nativas de un hardware) y `MAPPED` (mapeado a un dispositivo).

Punto de partida: **repositorio git inicializado y vacío**.

### Decisiones fijas (no cambiar sin preguntar)

- Lenguaje: **Python 3.13**. Circuitos con **Qiskit**; formato de intercambio **OpenQASM 3**.
- Conjunto de puertas canónico: **`rz`, `sx`, `x`, `cx`**.
- Las medidas **finales** se eliminan (`remove_final_measurements`). Los circuitos con medidas a mitad de circuito o control clásico (`dynamic_qft`, `ghz_dynamic`) quedan **fuera del núcleo**.
- Rango de tamaños por defecto: **2 a 40 cúbits** (configurable).
- Todo es reproducible: semillas en configuración, versiones fijadas, ningún paso manual.
- Los nombres de carpetas y scripts van en español, sin tildes ni eñes; el código y los comentarios, en español o en inglés, pero de forma consistente.

---

## 1. Fase 1: esqueleto del repositorio

### Tareas

1. Crear esta estructura:

```
.
├── configs/
│   ├── dataset.yaml          # algoritmos, familias, tamaños, niveles, variantes, semillas
│   └── splits.yaml           # proporciones y protocolos de partición
├── data/                     # en .gitignore salvo data/README.md
│   ├── raw/                  # F1: QASM tal y como sale de la fuente
│   ├── canonical/            # F2: QASM aplanado y normalizado
│   ├── features/             # F3
│   ├── graphs/               # F4 y F5
│   ├── tokens/               # F6
│   ├── gate_labels/          # F7
│   ├── splits/
│   └── metadata.csv
├── src/qcclass/              # paquete instalable
│   ├── __init__.py
│   ├── config.py             # carga y valida los YAML
│   ├── data/
│   │   ├── generar_mqt.py
│   │   ├── oraculos.py
│   │   ├── normalizar.py
│   │   ├── deduplicar.py
│   │   ├── particiones.py
│   │   └── calidad.py
│   ├── formatos/
│   │   ├── caracteristicas.py    # F3
│   │   ├── grafo_dag.py          # F4
│   │   ├── grafo_cubits.py       # F5
│   │   ├── tokens.py             # F6
│   │   └── etiquetas_puerta.py   # F7
│   └── cli.py                # punto de entrada único
├── tests/
├── notebooks/
├── results/                  # en .gitignore
├── pyproject.toml
├── Makefile
├── README.md
└── .gitignore
```

2. `pyproject.toml` con el paquete `qcclass` instalable en modo editable (`pip install -e .[dev]`) y estas dependencias: `qiskit`, `qiskit-qasm3-import`, `mqt.bench`, `numpy`, `pandas`, `pyyaml`, `networkx`, `scikit-learn`, `torch`, `torch-geometric`, `tqdm`, `typer` (CLI). Extras `dev`: `pytest`, `ruff`. Tras instalar, generar `requirements.lock` con `pip freeze` para fijar versiones.
3. `cli.py` con `typer` y estos subcomandos (de momento vacíos, que impriman "no implementado"): `generar`, `oraculos`, `normalizar`, `deduplicar`, `formatos`, `particiones`, `calidad`, `todo`.
4. `Makefile` con objetivos `install`, `lint`, `test`, `dataset` (ejecuta `qcclass todo`) y `clean-data`.
5. `README.md` breve: objetivo, instalación, cómo regenerar el dataset.

### Comprobación

- `pip install -e .[dev]` funciona, `qcclass --help` lista los subcomandos, `pytest` pasa (aunque haya un solo test trivial) y `ruff check .` no da errores.

---

## 2. Fase 2: configuración

`configs/dataset.yaml` debe contener, como mínimo:

```yaml
semilla_global: 42
tamanos: {min: 2, max: 40}
niveles: [alg, indep, nativegates, mapped]
nativegates: {gateset: ibm_falcon}      # comprobar nombres válidos en la versión instalada
mapped: {dispositivo: ibm_falcon_27}    # idem
opt_level: 2
variantes_compilacion: 3                # semillas de transpilación distintas por circuito base
variantes_parametros: 5                 # instancias con parámetros aleatorios (solo variacionales)
max_nodos: 20000                        # descartar circuitos con más puertas

familias:
  oraculos: [dj, bv, simon, grover]
  qft_fase: [qft, qftentangled, qpeexact, qpeinexact, iqpe, ae, mlqae, shor, hhl]
  aritmetica: [cdkm_ripple_carry_adder, vbe_ripple_carry_adder, draper_qft_adder,
               modular_adder, multiplier, hrs_cumulative_multiplier, rg_qft_multiplier,
               full_adder, half_adder]
  variacionales: [qaoa, vqe_real_amp, vqe_su2, vqe_two_local, qnn,
                  bmw_quark_cardinality, bmw_quark_copula]
  preparacion_estados: [ghz, wstate, graphstate, bell]
  simulacion: [qwalk, quantum_metropolis]
  correccion_errores: [seven_qubit_steane_code, shors_nine_qubit_code]
  protocolos: [teleportation, superdense_coding]

control_negativo: [randomcircuit]       # nunca es clase; se usa como "desconocido"
excluidos: [dynamic_qft, ghz_dynamic]
variacionales_con_parametros: [qaoa, vqe_real_amp, vqe_su2, vqe_two_local, qnn]
```

`config.py` carga el YAML, valida que **cada nombre de benchmark existe en la versión instalada de MQT Bench** (obtén la lista desde la propia librería, no la copies a mano) y avisa de los que no existan.

`configs/splits.yaml`:

```yaml
semilla: 42
aleatorio: {train: 0.70, val: 0.15, test: 0.15}
por_tamano: {train: [3, 16], val: [17, 20], test: [21, 30]}
algoritmos_reservados: [simon, qwalk, hhl]   # nunca se ven en entrenamiento
minimo_por_clase_train: 300
```

### Comprobación

- Test que carga ambos YAML, y test que falla si se introduce un benchmark inexistente.

---

## 3. Fase 3: generación desde MQT Bench (F1)

Implementar `generar_mqt.py`:

1. Para cada algoritmo, cada tamaño del rango y cada nivel, llamar a `get_benchmark(benchmark=..., level=BenchmarkLevel.X, circuit_size=n, ...)`. Para `NATIVEGATES` usar `target=get_target_for_gateset(<gateset>, n)`; para `MAPPED`, `target=get_device(<dispositivo>)` y saltar los tamaños que superen los cúbits del dispositivo. Pasar `opt_level` de la configuración.
2. Si un tamaño no es válido para un algoritmo, capturar la excepción, registrarla en `data/logs/generacion_errores.csv` y continuar.
3. Para los algoritmos de `variacionales_con_parametros`, generar `variantes_parametros` instancias (los parámetros son aleatorios por defecto en MQT Bench). Comprobar si la librería acepta semilla; si no la acepta, fijar `numpy.random.seed` y `random.seed` antes de cada llamada y documentarlo.
4. Guardar cada circuito en OpenQASM 3 en `data/raw/<nivel>/<familia>/<algoritmo>/<algoritmo>_n<n>_p<variante>.qasm`.
5. Añadir una fila por archivo a `data/metadata.csv` con las columnas: `id`, `algoritmo`, `familia`, `nivel`, `n_qubits`, `variante_param`, `variante_comp`, `semilla`, `origen` (`mqt`), `group_id`, `ruta_raw`.
   - `group_id` = `<algoritmo>_n<n>_p<variante_param>`. Todas las versiones derivadas del mismo circuito base (otros niveles, otras compilaciones) comparten `group_id`.
6. La generación es **incremental e idempotente**: si el archivo ya existe, no se regenera (salvo `--forzar`).

### Comprobación

- Un comando de prueba rápida (`qcclass generar --algoritmos ghz,qft --max 6`) que termine en segundos.
- Informe por consola: circuitos generados por algoritmo y nivel, y errores.
- Test: un circuito generado se vuelve a cargar con `qiskit.qasm3.loads` y tiene el número de cúbits esperado.

---

## 4. Fase 4: oráculos propios (ampliación del dataset)

MQT Bench ofrece pocas variantes por tamaño de los algoritmos de oráculo. Implementar en `oraculos.py` generadores propios **solo con puertas estándar de Qiskit** (sin depender de `tweedledum` ni de `PhaseOracle`):

| Algoritmo | Qué se aleatoriza | Notas |
|---|---|---|
| `dj` | Función constante o balanceada; para las balanceadas, la elección de CNOT y X | 50 % constantes, 50 % balanceadas |
| `bv` | Cadena secreta `s` de n bits | Evitar `s = 0…0` |
| `simon` | Cadena secreta `s` y construcción del oráculo 2-a-1 | Seguir la construcción estándar |
| `grover` | Uno o varios estados marcados; número de iteraciones óptimo | Oráculo de fase con MCX y X; difusor estándar |

Requisitos:

- Cada circuito se construye **con subcircuitos con nombre** (`oraculo`, `difusor`, `hadamard_inicial`, etc.) agregados como instrucciones, para que la fase de etiquetas por puerta (F7) pueda recuperar el bloque de origen.
- Se guardan en `data/raw/alg/oraculos/<algoritmo>/<algoritmo>_n<n>_o<semilla>.qasm` con `origen = propio` en `metadata.csv` y `group_id` propio.
- Una columna `oraculo_id` en `metadata.csv` identifica el oráculo, para poder crear particiones de oráculos no vistos.
- Número de oráculos por tamaño: configurable (por defecto, 20).

### Comprobación

- Test de corrección **por simulación** para tamaños pequeños (n ≤ 6) con `qiskit.quantum_info.Statevector`: DJ distingue constante y balanceada, BV recupera `s`, Grover amplifica el estado marcado por encima de 0,5 de probabilidad, y Simon produce cadenas ortogonales a `s`.

---

## 5. Fase 5: aplanado, normalización y deduplicación (F2)

### `normalizar.py`

Para cada circuito de `data/raw/`:

1. Cargar con `qasm3.loads`.
2. Eliminar medidas finales y barreras.
3. Transpilar con `basis_gates=["rz", "sx", "x", "cx"]`, `optimization_level=1` y `seed_transpiler=<semilla>`. Repetir con `variantes_compilacion` semillas distintas cuando el circuito venga de `alg` o `indep`. Los circuitos de `nativegates` y `mapped` se traducen igualmente al conjunto canónico.
4. Verificar que no queda ninguna definición `gate` ni ninguna puerta fuera del conjunto canónico.
5. Descartar los circuitos con más de `max_nodos` puertas y registrarlos.
6. Guardar en `data/canonical/<nivel>/<familia>/<algoritmo>/<id>_c<variante_comp>.qasm` y añadir `ruta_canonical`, `n_puertas`, `profundidad` a los metadatos (una fila por variante de compilación; `group_id` heredado).

### `deduplicar.py`

1. Calcular dos huellas por circuito canónico: SHA-256 del texto QASM normalizado (sin comentarios ni espacios redundantes) y hash Weisfeiler-Lehman (`networkx.weisfeiler_lehman_graph_hash`) del grafo DAG con el tipo de puerta como atributo de nodo.
2. Marcar como duplicados los circuitos con huellas idénticas. Si dos duplicados tienen **etiquetas distintas**, no borrar ninguno: listarlos en `data/logs/duplicados_conflictivos.csv`, porque es un hallazgo que hay que revisar a mano.
3. Columna `duplicado_de` en metadatos; los duplicados se excluyen de las fases siguientes.

### Comprobación

- Test: un circuito con una puerta `h` sale como `rz`/`sx` y es equivalente al original (`Operator(a).equiv(Operator(b))` para n ≤ 5).
- Informe: circuitos únicos por clase y nivel tras deduplicar.

---

## 6. Fase 6: formatos de entrada (F3 a F7)

Todos los formatos se derivan **del circuito canónico (F2)**, salvo F7, que necesita también el circuito algorítmico.

### F3. Vector de características (`caracteristicas.py`)

Un `data/features/features.parquet` con una fila por circuito y estas columnas: `n_qubits`, `profundidad`, `n_puertas`, recuento y proporción de cada puerta del conjunto canónico, `n_cx`, proporción de puertas de dos cúbits, CNOT por capa (media y máximo), y del grafo de interacción de cúbits: grado medio, densidad, diámetro (0 si no es conexo) y número de componentes.

### F4. Grafo DAG de puertas (`grafo_dag.py`)

- Nodos = puertas (`dag.op_nodes()` de `qiskit.converters.circuit_to_dag`). Aristas = dependencia directa entre dos puertas que actúan sobre el mismo cúbit, dirigidas en el sentido temporal.
- Atributos de nodo: one-hot del tipo de puerta; `sin(θ)` y `cos(θ)` (0 y 1 si la puerta no tiene ángulo); indicador de control y de destino en `cx`; capa / profundidad (posición relativa); índice de cúbit / n_qubits.
- Salida: `torch_geometric.data.Data(x, edge_index, y_algoritmo, y_familia, id)` guardado en `data/graphs/dag/<id>.pt`, más un `InMemoryDataset` o una lista indexada para cargar todo de una vez.
- El orden de los nodos debe ser **determinista** (orden topológico estable) y guardarse, porque F7 depende de él.

### F5. Grafo de interacción de cúbits (`grafo_cubits.py`)

Nodos = cúbits; arista entre dos cúbits que comparten al menos una `cx`, con peso = número de `cx`. Atributos de nodo: número de puertas de un cúbit sobre él y grado. Mismo formato de salida en `data/graphs/cubits/`.

### F6. Secuencia de tokens (`tokens.py`)

Una línea por circuito en `data/tokens/tokens.jsonl` con `id` y la lista de tokens en el orden de F4, por ejemplo `["rz q0 0.785", "sx q0", "cx q0 q3"]`. Ángulos redondeados a 3 decimales. Generar también el vocabulario.

### F7. Etiquetas por puerta (`etiquetas_puerta.py`)

Objetivo: asignar a cada nodo de F4 el bloque de alto nivel del que procede.

1. Solo se aplica a circuitos con estructura de bloques: nivel `alg` de MQT Bench y oráculos propios.
2. Recorrer las instrucciones de primer nivel del circuito algorítmico. Descomponer **cada instrucción por separado** al conjunto canónico con `optimization_level=0`, y concatenar los resultados anotando, para cada puerta, el nombre de la instrucción de origen.
3. El circuito resultante de esa concatenación se guarda como una variante canónica adicional (`variante_comp = etiquetada`) para que **las etiquetas y el grafo F4 correspondan exactamente al mismo circuito**.
4. Salida: `data/gate_labels/<id>.json` con la lista de etiquetas en el orden de los nodos de F4 y el diccionario de bloques encontrados.

### Comprobación

- Para 5 circuitos al azar: número de nodos de F4 = número de puertas de F2 = número de tokens de F6 = número de etiquetas de F7 (cuando exista).
- Test: un circuito de 2 cúbits conocido produce exactamente el grafo y los atributos esperados.
- Notebook `notebooks/01_inspeccion.ipynb` que dibuja un circuito, su DAG coloreado por las etiquetas F7 y su grafo de cúbits.

---

## 7. Fase 7: particiones (`particiones.py`)

Generar en `data/splits/` un archivo por protocolo, cada uno con columnas `id` y `conjunto` (`train`, `val`, `test`, `desconocido`):

1. `aleatorio.csv`: 70/15/15 **por `group_id`** (nunca dos variantes del mismo grupo en conjuntos distintos), estratificado por algoritmo, con `StratifiedGroupKFold` o `GroupShuffleSplit` de scikit-learn.
2. `por_tamano.csv`: según los rangos de `splits.yaml`.
3. `oraculos_no_vistos.csv`: en la familia de oráculos, los `oraculo_id` de test no aparecen en train.
4. `por_nivel.csv`: train en `indep`, test en `nativegates` y `mapped`.
5. `abierto.csv`: los `algoritmos_reservados` y el control negativo, todos en `desconocido`; el resto, como en `aleatorio.csv`.

Las particiones se crean **una sola vez** con la semilla y no se regeneran salvo `--forzar`.

### Comprobación

- Test que garantiza que ningún `group_id` aparece en dos conjuntos en ningún protocolo.
- Test que garantiza que ningún algoritmo reservado aparece en train.
- Aviso si alguna clase queda por debajo de `minimo_por_clase_train`.

---

## 8. Fase 8: control de calidad y pipeline completo

`calidad.py` genera `results/informe_dataset.md` con:

- Circuitos únicos por clase, familia y nivel, antes y después de deduplicar.
- Distribución de cúbits y profundidad por clase (tabla y figura).
- Clases por debajo del mínimo.
- Errores de generación y duplicados conflictivos.
- Comprobación de coherencia entre `metadata.csv` y los archivos en disco.

`qcclass todo` ejecuta en orden: generar → oráculos → normalizar → deduplicar → formatos → particiones → calidad. `make dataset` lo lanza.

### Comprobación

- `make dataset` con una configuración reducida (`configs/dataset_mini.yaml`: 4 algoritmos, 3–8 cúbits) termina sin errores en pocos minutos. Añadir esta ejecución como test de integración marcado con `@pytest.mark.slow`.

---

## 9. Fases posteriores (no empezar hasta que lo indique)

10. **Línea base:** Random Forest y XGBoost sobre F3; experimento de control con solo `n_qubits` y `profundidad`.
11. **GNN:** PyTorch Geometric sobre F4, con 3–5 capas GIN o GraphSAGE, *pooling* media + máximo, dos cabezas (familia y algoritmo), entropía cruzada ponderada, Adam con lr 1e-3, parada temprana y 5 semillas por configuración.
12. **Ablaciones:** F5 y F6 frente a F4; aristas dirigidas frente a no dirigidas; con y sin ángulos.
13. **Explicabilidad:** GNNExplainer y atención (GATv2); comparar subgrafos relevantes con F7 (precisión, exhaustividad, IoU).
14. **Desconocidos:** umbral de probabilidad y distancia a centroides sobre el protocolo `abierto`.
15. **Seguimiento de experimentos** con MLflow, una configuración YAML por experimento.

---

## 10. Reglas generales para la IA

- No hagas ningún commit los hago yo.
- Funciones pequeñas, con anotaciones de tipo y docstring; nada de rutas escritas a mano fuera de la configuración y toda la documenatción en inglés.
- Registrar con `logging`, no con `print`, salvo en la CLI.
- No descargar nada de la web de MQT Bench: todo se genera con el paquete.
- Si un paso tarda mucho, añadir barra de progreso (`tqdm`) y paralelismo opcional (`--procesos N`).
- Si una decisión de este plan resulta imposible o claramente mejorable al ver la API real, **detente y pregúntame**.
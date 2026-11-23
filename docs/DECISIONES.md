# Decisiones del proyecto

Registro vivo de cómo se trabaja en este repositorio y de las decisiones que se apartan de
`PLAN.md` o lo concretan. Se actualiza en cada fase, en el mismo commit que el código.

## Forma de trabajo

- **Fases.** Se sigue `PLAN.md` fase a fase. Al terminar una fase se ejecutan sus
  comprobaciones y tests, se presenta un resumen y no se empieza la siguiente sin
  confirmación explícita.
- **Commits.** Solo los hace el responsable del proyecto. Cada commit de fase incluye la
  actualización de este archivo.
- **APIs reales.** Antes de usar una librería se comprueba la versión instalada
  (`pip show`, `inspect`, `help`). Si el plan no encaja con la API real, se para, se avisa y
  se propone una alternativa; no se improvisa.
- **Tests.** Cada fase añade tests con `pytest`; `make test` excluye los marcados con
  `@pytest.mark.slow` y `make test-all` los incluye. `ruff check` y `ruff format --check`
  deben pasar. Los tests nunca escriben en `data/`: usan la fixture `make_config`, que
  copia la configuración a un directorio temporal.
- **Logging.** Módulos con `logging.getLogger(__name__)`; `print`/`typer.echo` solo en la
  CLI. El logger raíz queda en WARNING (Qiskit registra cada pase del transpilador en INFO)
  y los loggers `qcclass` en INFO, o DEBUG con `--verbose`.
- **Rutas.** Ninguna ruta escrita a mano en el código: todas salen de la sección `rutas`
  de `configs/dataset.yaml` (`raiz` relativa al YAML; el resto, relativas a `raiz`). En los
  metadatos las rutas se guardan relativas a `raiz`.
- **Semillas.** Todo aleatorio sale de `semilla_global` mediante
  `qcclass.semillas.derive_seed` (SHA-256 de la semilla y de identificadores estables; no se
  usa `hash()`). A MQT Bench se le pasa siempre una semilla explícita.
- **Reproducibilidad.** Python 3.13, versiones fijadas en `requirements.lock`
  (`make install` instala con ellas como restricciones). La generación es incremental e
  idempotente; `--forzar` regenera.
- **Idioma.** Carpetas y scripts en español sin tildes ni eñes; código, comentarios y
  documentación técnica en inglés. Este archivo está en español por ser el registro de
  trabajo con el responsable del proyecto.

## Registro de decisiones

| Fecha | Fase | Decisión | Motivo | Alternativa descartada |
|---|---|---|---|---|
| 2026-11-22 | 1 | Python 3.13 instalado con Homebrew y entorno `.venv` | Solo había Python 3.15 y 3.9 en la máquina | Usar Python 3.15 (sin wheels garantizadas de torch/torch-geometric) |
| 2026-11-22 | 1 | Dependencia `pyarrow` añadida | pandas la necesita para escribir el Parquet de F3 | — |
| 2026-11-22 | 1 | `.gitignore` con patrones anclados (`/data/*`) | `data/` a secas ignoraría también `src/qcclass/data/` | — |
| 2026-11-22 | 2 | Gateset `ibm_heron` y dispositivo `ibm_heron_156` | `ibm_falcon` e `ibm_falcon_27` no existen en mqt.bench 2.4.0; Heron es heavy-hex como Falcon y cubre 2–40 cúbits | Registrar un Falcon propio con mapa de acoplamiento escrito a mano; Nighthawk |
| 2026-11-22 | 2 | Sección `rutas` en `dataset.yaml` y validación de nombres contra la librería instalada | Regla de no escribir rutas a mano; evitar listas copiadas | — |
| 2026-11-23 | 3 | Semilla explícita por circuito base, compartida por todos sus niveles | Sin semilla, MQT Bench 2.4 usa siempre `seed=10` y las variantes saldrían idénticas | `numpy.random.seed` global |
| 2026-11-23 | 3 | Excluidos `iqpe`, `mlqae`, `quantum_metropolis`, `seven_qubit_steane_code`, `shors_nine_qubit_code`, `teleportation` (además de `dynamic_qft`, `ghz_dynamic`), más un detector automático de circuitos dinámicos | Tienen medidas intermedias, `reset` o control clásico en 2.4.0 | Convertirlos a unitarios con medida diferida |
| 2026-11-23 | 3 | Excluido `shor` | Su único tamaño válido en 2–40 es 18, con ~55 000 puertas en nativegates (> `max_nodos`) | Subir `max_nodos` |
| 2026-11-23 | 3 | Sonda nativegates: si la versión nativegates de una base supera `max_nodos`, no se genera ningún nivel y se dejan de subir tamaños | Grover y qwalk explotan (25 M de puertas con n=24) y gastarían GB y horas para descartarse en F2 | Generar todo con un tope duro y filtrar en F2 |
| 2026-11-23 | 3 | Envolver instrucciones con parámetros matriciales (`c-unitary` de `ae`) en puertas con nombre equivalentes antes de exportar a QASM 3 | `qasm3.dumps` no puede escribirlas | Dejar `ae` sin niveles alg/indep |
| 2026-11-23 | 3 | `n_qubits` = anchura lógica; los circuitos `mapped` se guardan con los 156 cúbits del dispositivo | raw guarda el circuito tal y como sale; los cúbits inactivos se quitan en F2 | — |
| 2026-11-23 | 3 | Cada circuito de raw se guarda también en QPY; las fases 5 y 6 leen QPY | `qasm3.loads` va a ~900 puertas/s y renombra las puertas envueltas; QPY es ~10⁴ veces más rápido y conserva nombres y anchura | Leer QASM 3 / usar `loads_experimental` (falla con expresiones `pi/2`) |
| 2026-11-23 | 3 | `minimo_por_clase_train` se mide en `group_id` únicos y vale 15 | Un circuito base y sus variantes no son muestras independientes | 300 archivos |
| 2026-11-23 | 3 | `algoritmos_reservados: [qwalk, hrs_cumulative_multiplier, hhl]`; `simon` pasa a clase normal | Decisión del responsable | `[simon, qwalk, hhl]` del plan |
| 2026-11-23 | 3 | Eliminada la familia `simulacion`; `qwalk` se genera en `reservados_sin_familia` (`familia = sin_familia`) | Solo quedaba `qwalk`, que está reservado | Moverlo a `control_negativo` u otra familia |
| 2026-11-23 | 3 | Ningún circuito se borra por desequilibrio: `max_bases_por_clase` se aplica a todas las clases en la Fase 7, por muestreo con semilla o por ponderación de la pérdida (`splits.yaml: balanceo`) | Conservar todos los datos y decidir el balanceo en las particiones | Recortar los variacionales en la generación (implementado y retirado) |
| 2026-11-23 | 4 | Oráculos propios con bloques con nombre y deduplicados por `oraculo_id`; DJ con solo 2 constantes por tamaño | Solo existen dos funciones constantes por tamaño; repetirlas daría duplicados | Forzar 50 % de constantes |
| 2026-11-23 | 4 | MCX v-chain construido a mano con `ccx` (k−2 ancillas limpias) | `QuantumCircuit.mcx(mode="v-chain")` está obsoleto desde Qiskit 2.1 | Usar el argumento obsoleto |
| 2026-11-23 | 4 | Grover mezcla dos MCX, mitad y mitad por tamaño con semilla: v-chain manual y `synth_mcx_n_clean_m15` (⌈(k−2)/2⌉ ancillas); columna `construccion_mcx` | Que la anchura y la escalera de `ccx` no delaten a Grover; admite tamaños pares | Solo v-chain (solo tamaños impares) |
| 2026-11-23 | 5 | Variantes de compilación por nivel de optimización: `niveles_opt_compilacion: [1, 0, 2]` (c0 = opt 1, como el plan); columna `opt_level_comp`; nativegates/mapped solo c0 | Sin mapa de acoplamiento `transpile` es determinista: con semillas distintas las 3 variantes salían idénticas | Rutado sobre una línea con semilla (añade SWAP y se parece a `mapped`); una sola variante |
| 2026-11-23 | 5 | Se acepta que con opt 2 Qiskit convierta SWAP en una permutación de salida (`final_layout`, guardada en el QPY) | Es la optimización del propio transpilador; la equivalencia se comprueba con `Operator.from_circuit` | Desactivar `ElidePermutations` |
| 2026-11-23 | 5 | Los circuitos `mapped` pierden sus cúbits inactivos al normalizar | Su anchura canónica pasa a ser la lógica y no delata el nivel | — |
| 2026-11-23 | 5 | F2 se guarda como QASM 3 legible más gemelo QPY; las filas canónicas se añaden a `metadata.csv` (`variante_comp` = c0, c1…) junto a las de raw (`variante_comp` vacío) | Una sola tabla, como pide el plan; QPY acelera las fases siguientes | Tabla canónica separada |
| 2026-11-23 | 5 | Solo excluyen los duplicados exactos (SHA-256 del QASM normalizado, misma etiqueta); el hash WL se guarda y sus coincidencias entre etiquetas distintas se listan como `estructural` | WL ignora ángulos y etiquetas de cúbit: borraría las variantes paramétricas y las permutaciones | Excluir también por WL |
| 2026-11-23 | 5 | Representante de cada grupo de duplicados: el de nivel más alto (alg < indep < nativegates < mapped) y, a igualdad, el menor `id` | Determinista y conserva preferentemente la versión algorítmica | — |
| 2026-11-23 | 5 | Propuesta de fusiones / etiqueta solo de familia en `results/informe_deduplicacion.md`, sin aplicarla | Decisión aplazada por el responsable hasta ver recuentos deduplicados | Decidirlo antes de deduplicar |
| 2026-11-23 | 5 | Al escribir QPY se comprueba que se puede leer; si no, las puertas de librería se envuelven en puertas con nombre equivalentes (`make_qpy_safe`) | Qiskit 2.6 escribe `FullAdderGate`, `HalfAdderGate`, `ModularAdderGate` y `MultiplierGate` en QPY pero no puede leerlas (falta `num_state_qubits`) | Leer esos circuitos desde QASM 3 |
| 2026-11-23 | 5 | Fusiones `full_adder` → `cdkm_ripple_carry_adder` y `bell` → `ghz` (`etiquetas.fusiones`, columna `clase`) | full_adder y cdkm coinciden en el 95 % de sus bases; bell/ghz por decisión del responsable (solo 5 % idénticas) | Mantenerlas separadas |
| 2026-11-23 | 5 | `draper_qft_adder` y `modular_adder` no se fusionan | Solo el 10 % de sus bases son idénticas (2 de 20; n = 6 y 8) | Fusionarlas |
| 2026-11-23 | 5 | `vbe_ripple_carry_adder` es clase propia solo con ≥ 22 bases útiles; si no, solo etiqueta de familia (`clase` vacía). Hoy tiene 13 → solo familia | No se parece a ninguna otra clase (8 %) y no llega al mínimo | Fusionarla con full_adder |
| 2026-11-23 | 5 | La propuesta de fusión exige que más del 50 % de las bases de la clase más pequeña sean idénticas a la otra | La regla anterior proponía fusiones con un solo grupo compartido | Partner = clase con más grupos compartidos |
| 2026-11-23 | 5 | Los duplicados exactos con etiquetas distintas se excluyen de todas las clases implicadas (`exclusion = conflicto_etiquetas`) y se listan en el informe | Un mismo circuito no puede tener dos etiquetas de entrenamiento | Conservarlos y solo listarlos |
| 2026-11-23 | 4 | DJ balanceado: 70 % no lineales `f(x) = x_i ⊕ g(resto)` (g = XOR de 1–3 productos de 2–3 variables, con CCX o MCX de las tres construcciones) y 30 % lineales; columna `tipo_oraculo_dj` | Una paridad con X envolventes sigue siendo BV disfrazado | Exigir X envolventes en las paridades |
| 2026-11-23 | 4 | Tercera construcción MCX `b95` (`synth_mcx_1_clean_b95`); las construcciones que caben en cada tamaño se reparten a partes iguales con semilla | Más tamaños con al menos dos construcciones; existen Grover de todos los tamaños ≥ 2 | Solo v-chain + m15 |
| 2026-11-23 | 7 | Protocolo `por_nivel`: train en ALG + INDEP (deduplicados entre sí), test en NATIVEGATES + MAPPED (`splits.yaml: por_nivel`) | INDEP casi duplica ALG tras normalizar y dejaba el train vacío | Train solo en INDEP |
| 2026-11-23 | 6 | Orden de nodos de F4/F6/F7 = orden de `circuit.data` del circuito canónico (orden topológico estable, implícito: nodo i = instrucción i) | Es determinista y permite alinear las etiquetas F7 sin guardar un mapeo aparte | `dag.topological_op_nodes()` con mapeo guardado |
| 2026-11-23 | 6 | F4: el papel de control/destino de cada cúbit en una CX va en los atributos de arista (`edge_attr` = control/destino en origen y en destino); los nodos llevan el índice relativo de sus dos cúbits | Un nodo CX es a la vez control y destino: el indicador por nodo no distingue nada | Indicadores por nodo |
| 2026-11-23 | 6 | F4/F5 se guardan como un `.pt` por circuito (diccionario de tensores, carga con `weights_only=True`) + `indice.csv` + `IndexedGraphDataset` de carga perezosa | ~14 M de nodos (1–1,5 GB): un `InMemoryDataset` único no cabe cómodamente en memoria | `InMemoryDataset` colacionado |
| 2026-11-23 | 6 | F6: ángulos reducidos a [-π, π) y redondeados a 3 decimales; sin "-0.000" | Rotaciones equivalentes comparten token | Ángulo tal cual |
| 2026-11-23 | 6 | Variante `etiquetada` (F7) con `exclusion = etiquetada`: tiene todos los formatos pero nunca entra en las particiones | Es auxiliar para explicabilidad; con opt 0 suele coincidir con c1 | Tratarla como variante de entrenamiento |
| 2026-11-23 | 6 | Etiquetas de F4/F5: `y_algoritmo`/`y_familia` = índices de `data/graphs/etiquetas.json`; -1 para el control negativo y los reservados sin familia, y `y_algoritmo = -1` en las clases que solo conservan la familia | Un solo vocabulario compartido por todos los formatos | Etiquetas como texto |
| 2026-11-23 | 6 | Dependencias nuevas: `matplotlib`, `pylatexenc`; extra dev `nbformat`, `nbclient`, `ipykernel`. El notebook se ejecuta en un test `slow` | El notebook de inspección las necesita | Notebook sin ejecución automática |
| 2026-11-23 | 6 | Solo etiqueta de familia para `bmw_quark_copula`, `draper_qft_adder`, `half_adder`, `modular_adder`, `multiplier` y `rg_qft_multiplier` (`etiquetas.solo_familia`); draper y modular no se fusionan. Quedan 22 etiquetas de algoritmo: 20 entrenables y 2 reservadas (`hhl`, `hrs_cumulative_multiplier`) | Menos de 22 bases útiles; solo 10 % de bases idénticas entre draper y modular | Fusionar draper y modular |
| 2026-11-23 | 6 | Códigos: `y_algoritmo = -1` sin etiqueta de algoritmo (la cabeza de familia sí lo usa); `-2` desconocido (control negativo y reservados, solo en evaluación). Los reservados con familia conservan `y_familia`; `qwalk` y `randomcircuit` llevan `y_familia = -2`. `etiquetas.json` documenta los códigos | Evaluar si la cabeza de familia generaliza a algoritmos no vistos | Un único código -1 |
| 2026-11-23 | 7 | La variante `etiquetada` sigue a su `group_id`: en val/test entra con `solo_evaluacion = True`; en train se excluye. La deduplicación la ignora | Material de evaluación para explicabilidad sin contaminar train | Excluirla de todas las particiones |
| 2026-11-23 | 7 | Asignación estratificada por etiqueta: los `group_id` de cada etiqueta se reparten 70/15/15 con `GroupShuffleSplit` en dos etapas (semilla derivada por protocolo y etiqueta); con menos de 3 grupos, uno a train y otro a test | Estratificación exacta y ningún grupo en dos conjuntos | `StratifiedGroupKFold` con 20 pliegues |
| 2026-11-23 | 7 | Balanceo por muestreo solo en train (máximo `max_bases_por_clase` grupos por etiqueta, por defecto la mediana = 39); val y test completos; los grupos no muestreados no aparecen en el CSV pero siguen en disco | Conservar toda la evaluación; las métricas deben ser macro | Muestrear antes de partir (reduce también test) |
| 2026-11-23 | 7 | `por_nivel`: train = niveles ALG+INDEP de los grupos de train; val = esos niveles de los grupos de val; test = NATIVEGATES+MAPPED de otros grupos | Un mismo grupo no puede estar en train y test | Mismo grupo en train (alg) y test (mapped) |
| 2026-11-23 | 7 | `oraculos_no_vistos`: unidad de partición `oraculo_id` (`mqt:<group_id>` para los de MQT) | Ningún oráculo de test se ve en train | — |
| 2026-11-23 | 7 | `abierto` = `aleatorio` (misma asignación y mismo muestreo) + desconocidos; `aleatorio` no incluye desconocidos | El plan define `abierto` como `aleatorio` más los desconocidos | — |

## Avisos pendientes

- **DJ.** Hay 78 constantes, 229 lineales y 447 no lineales (los tamaños 2 y 3 solo
  admiten lineales). 45 grupos DJ lineal = BV se excluyen de ambas clases.
- **Desequilibrio de clases.** Bases útiles entre 5 (`hrs_cumulative_multiplier`) y 778
  (`dj`). Se resolverá en la Fase 7 con `balanceo` (muestreo o ponderación).
- **Propuesta pendiente de decisión** (`results/informe_deduplicacion.md`): solo etiqueta
  de familia para `bmw_quark_copula`, `draper_qft_adder`, `half_adder`, `modular_adder`,
  `multiplier` y `rg_qft_multiplier` (menos de 22 bases útiles); ninguna fusión nueva supera
  el 50 %.
- **Grover: tamaños con una sola construcción.** En 4, 6, 10, 16, 22… solo cabe una de las
  tres. Además, los descartes por `max_nodos` rompen el equilibrio en los tamaños grandes
  (b95 usa más cúbits de búsqueda y supera antes el límite): Grover propio llega a n = 13.
- **F7 y renombrados (resuelto).** F7 lee QPY y conserva los nombres originales de los
  bloques (`c-unitary`, `qft_dg`, `oraculo`…); comprobado en los tests.
- **INDEP casi duplica ALG** tras normalizar; por eso `por_nivel` entrena en ALG + INDEP.
- **Modelos (Fase 11): usar `edge_attr`.** El papel de control/destino de cada cúbit en
  una CX solo está en `edge_attr` de F4: hay que usar capas que lo consuman (GINEConv o
  GATv2 con `edge_dim`). GIN o GraphSAGE puros lo ignorarían.
- **Evaluación: desgloses obligatorios.** Desglosar el acierto de `grover` por
  `construccion_mcx` y el de `dj` por `tipo_oraculo_dj`, para detectar si el modelo se apoya
  en la construcción o en la linealidad del oráculo.
- **Por debajo del mínimo de train.** `familia:variacionales` (solo `bmw_quark_copula`) tiene
  14 grupos de train (< 15) en `aleatorio`, `por_nivel` y `abierto`; en `por_tamano` lo están
  11 etiquetas, porque el rango de train (3–16 cúbits) deja pocas bases por clase.
- **`por_tamano` sin Grover en test.** Los Grover acaban en n = 13, así que no hay Grover en
  test (21–30 cúbits) ni F7 de Grover en ese protocolo.


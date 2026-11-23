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

## Avisos pendientes

- **DJ 78/676.** Las constantes son 78 frente a 676 balanceadas (solo hay 2 funciones
  constantes distintas por tamaño).
- **Desequilibrio de clases.** Entre 5 bases (`hrs_cumulative_multiplier`) y 793 (`dj`),
  mediana 39. Se resolverá en la Fase 7 con `balanceo`; la propuesta de fusionar clases o
  dejarlas solo con etiqueta de familia irá en el informe de calidad tras la Fase 5, con
  recuentos deduplicados.
- **F7 y renombrados.** Al recargar QASM 3, el exportador renombra las puertas envueltas
  (`_c_unitary_0`, …) y las permutaciones (`permutation__3_2_1_0_`). Mitigado leyendo QPY,
  que conserva los nombres originales; hay que comprobarlo al implementar F7.
- **Grover: tamaños sin construcción.** Con las dos construcciones no hay circuito de
  4, 10, 16, 22… cúbits, y en 6, 7, 8, 12, 13, 14… solo cabe una de las dos, así que en esos
  tamaños la anchura sigue delatando la construcción.
- **DJ balanceado = BV.** 66 grupos de circuitos canónicos idénticos entre `dj` y `bv`: un
  oráculo DJ balanceado sin X envolventes es exactamente BV con `s = S`. No se borran (etiquetas
  distintas); se puede evitar generando solo DJ balanceados con X o filtrando esos casos.
- **`full_adder` ≈ `cdkm_ripple_carry_adder`.** 18 grupos idénticos (el `full_adder` de MQT
  es la misma construcción CDKM). Candidatos claros a fusión.
- **INDEP casi duplica ALG.** Tras normalizar, la mayoría de los circuitos `indep` coinciden
  con su `alg` (en mqt.bench 2.4 INDEP transpila sin conjunto de puertas) y se marcan como
  duplicados; el protocolo `por_nivel` (train en `indep`) tendrá pocos datos.
- **Propuesta de clases pendiente de decisión** en `results/informe_deduplicacion.md`:
  fusiones en `aritmetica` (cdkm+full_adder, draper+modular_adder…), solo familia para
  `multiplier`, `rg_qft_multiplier`, `half_adder`, `bmw_quark_copula`; `bell` con `ghz`.
- **Grover en la Fase 3 (MQT).** Hasta n=9 por la sonda; los oráculos propios llegan a n=17.


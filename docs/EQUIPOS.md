# Equipos

Papel de cada equipo, rutas, red, configuración (`.env`), servidores y trabajos largos.
Documento vivo: se actualiza en cada fase. Las decisiones están en
[DECISIONES.md](DECISIONES.md).

## Resumen

| Equipo | Máquina | Papel | Repositorio | IP (red local) |
|---|---|---|---|---|
| `m3pro` | MacBook Pro M3 Pro, 18 GB | Desarrollo, línea base, WL, análisis; GNN en CPU si hace falta. Origen de los datos | `/Users/samuel/Developer/personal/QTagger` | 192.168.1.136 |
| `rtx3050` | Portátil Windows, RTX 3050 de 4 GB, WSL2 (Ubuntu, 11 GB) | Equipo principal de la GNN (búsqueda, ablaciones, modelos finales) | `/home/samue/QTagger` | 192.168.1.140 |
| `m1pro` | MacBook Pro M1 Pro, 16 GB | Servidor de MLflow y del almacenamiento de Optuna | `/Users/samuel/Documents/QTagger` | 192.168.1.133 |

- Código: se sincroniza con git (`git pull`; si cambia `requirements.lock`, `make install`).
- Datos: **no se regeneran**. Se copian desde el `m3pro` y se comprueban con
  `qcclass verificar` (huellas de `data/VERSION.json`, versión actual `dataset-v1.1`).
- MPS no se usa: en la medición resultó más lento que la CPU del M3 Pro.
- Estado (2026-11-24, fase 9): servidores en marcha en el `m1pro`; `ping`, `verificar` y la
  prueba de Optuna/MLflow correctos desde los tres equipos. Python 3.13.18 y las mismas
  versiones de librerías en `m3pro` y `rtx3050`.

## Red

| Servicio | Equipo | Puerto | URI de los clientes |
|---|---|---|---|
| MLflow (interfaz y API) | `m1pro` | 5050 | `http://192.168.1.133:5050` |
| Almacenamiento de Optuna (proxy gRPC) | `m1pro` | 13000 | `grpc://192.168.1.133:13000` |
| SSH (copia de datos con `rsync`) | `m3pro` | 22 | `samuel@192.168.1.136` |

- **IPs fijas.** Conviene reservar las tres IPs en el router (reserva DHCP por dirección MAC);
  si cambian, solo hay que actualizar los `.env`.
- **Puerto 5050 y no 5000**: en macOS el 5000 lo ocupa el receptor de AirPlay.
- **Cortafuegos del `m1pro`.** Si el cortafuegos de macOS está activo, la primera vez que se
  arrancan los servidores pide permitir conexiones entrantes a `Python`: hay que aceptarlo
  (o añadir `/Users/samuel/Documents/QTagger/.venv/bin/python` en *Ajustes del Sistema →
  Red → Cortafuegos → Opciones*).
- **WSL2 en modo espejo.** En `C:\Users\<usuario>\.wslconfig`, sección `[wsl2]`:
  `networkingMode=mirrored` (así, con esa ortografía). El `rtx3050` solo es cliente: le basta
  con poder salir hacia el `m1pro` y el `m3pro`.
- **SSH en el `m3pro`** (para que los otros equipos copien los datos): *Ajustes del Sistema →
  General → Compartir → Sesión remota* activada para el usuario `samuel`.

## Configuración por equipo (`.env`)

Cada equipo tiene un `.env` en la raíz del repositorio. **No se versiona** (está en
`.gitignore`); `.env.example` es la plantilla. Las variables del entorno tienen prioridad
sobre el archivo. Si falta una variable o un servidor no responde, los comandos abortan con
un mensaje claro: nunca registran en local.

`m3pro` y `rtx3050` (clientes; cambia solo `QCCLASS_EQUIPO`):

```bash
QCCLASS_EQUIPO=m3pro            # o rtx3050
QCCLASS_MLFLOW_URI=http://192.168.1.133:5050
QCCLASS_OPTUNA_STORAGE=grpc://192.168.1.133:13000
```

`m1pro` (servidor y, para comprobarse a sí mismo, cliente):

```bash
QCCLASS_EQUIPO=m1pro
QCCLASS_SERVIDORES_DIR=/Users/samuel/qcclass_servidores   # bases de datos, artefactos, logs
QCCLASS_SERVIDORES_HOST=0.0.0.0
QCCLASS_MLFLOW_PUERTO=5050
QCCLASS_OPTUNA_PUERTO=13000
QCCLASS_MLFLOW_URI=http://127.0.0.1:5050
QCCLASS_OPTUNA_STORAGE=grpc://127.0.0.1:13000
```

Comprobación desde cualquier equipo: `qcclass ping` (o `make ping`).

## Servidores (`m1pro`)

```bash
cd /Users/samuel/Documents/QTagger && source .venv/bin/activate
make servidores          # = qcclass servidores arrancar
make estado-servidores   # = qcclass servidores estado
make parar-servidores    # = qcclass servidores parar
```

- `arrancar` lanza dos procesos independientes de la terminal (se puede cerrar): el servidor de
  MLflow (base de datos SQLite y artefactos en `QCCLASS_SERVIDORES_DIR/mlflow/`, servidos por
  el propio servidor) y el proxy gRPC de Optuna (SQLite en `QCCLASS_SERVIDORES_DIR/optuna/`).
  Espera hasta que ambos responden. Si ya estaban en marcha, no los duplica.
- Interfaz de MLflow: <http://192.168.1.133:5050> desde cualquier equipo de la red.
- Logs: `QCCLASS_SERVIDORES_DIR/logs/{mlflow,optuna}.log`.
- Copia de seguridad: con los servidores parados, copiar `QCCLASS_SERVIDORES_DIR` entero.

### Evitar el reposo del `m1pro`

- `arrancar` lanza también `caffeinate -i -s -w <pid de MLflow>`: mientras viva MLflow, el Mac
  no entra en reposo por inactividad (y, con el cargador conectado, tampoco en reposo del
  sistema). Se apaga solo al parar los servidores.
- **Cerrar la tapa sí lo duerme** salvo en modo «tapa cerrada» (cargador, pantalla y teclado
  externos). Lo más sencillo: tapa abierta, cargador conectado y brillo al mínimo.
- Alternativa, si hace falta cerrar la tapa: `sudo pmset -a disablesleep 1` mientras duren
  los experimentos y `sudo pmset -a disablesleep 0` al terminar.

## Datos

El `m3pro` es el origen. Los otros equipos copian con `rsync` por SSH (solo transfiere lo que
cambia; hace falta la sesión remota del `m3pro`) y después comprueban con `qcclass verificar`.

`rtx3050` (todo menos `raw/` y `canonical/`, que solo hacen falta para regenerar), desde
`/home/samue/QTagger`:

```bash
rsync -av --exclude raw/ --exclude canonical/ --exclude logs/ \
  samuel@192.168.1.136:Developer/personal/QTagger/data/ data/
qcclass verificar
```

`m1pro` (no entrena: solo lo que cubren las huellas), desde `/Users/samuel/Documents/QTagger`:

```bash
mkdir -p data/graphs
rsync -av samuel@192.168.1.136:Developer/personal/QTagger/data/metadata.csv data/
rsync -av samuel@192.168.1.136:Developer/personal/QTagger/data/splits data/
rsync -av samuel@192.168.1.136:Developer/personal/QTagger/data/graphs/etiquetas.json data/graphs/
qcclass verificar
```

Tras cada versión nueva del dataset basta con repetir las mismas órdenes.

## Trabajos largos en el `rtx3050` (tmux)

Los entrenamientos de las fases 12, 13 y 15 se lanzan dentro de `tmux`, para que sigan
corriendo aunque se cierre la terminal. Instalación única: `sudo apt install -y tmux`.

```bash
cd /home/samue/QTagger && source .venv/bin/activate
mkdir -p logs                    # ignorada por git
# lanzar (sesión <nombre>, salida también en logs/<nombre>.log)
tmux new-session -d -s <nombre> "source .venv/bin/activate && <comando> 2>&1 | tee -a logs/<nombre>.log"
tmux ls                          # sesiones en marcha
tmux attach -t <nombre>          # entrar; salir sin pararla: Ctrl-b y luego d
tail -f logs/<nombre>.log        # progreso sin entrar en la sesión
watch -n 30 nvidia-smi           # uso de GPU, memoria y temperatura
tmux kill-session -t <nombre>    # parar el trabajo
```

El progreso también se ve en MLflow (métricas por época) y, durante una búsqueda, en el
estudio de Optuna. Los entrenamientos guardan un punto en cada época: si se cortan, al
relanzar el mismo comando se reanudan desde el último.

Condiciones del portátil Windows durante un trabajo largo:

- Cargador conectado; en *Configuración → Sistema → Energía*, «Suspender» en «Nunca» con el
  cargador conectado, y en el Panel de control, al cerrar la tapa, «No hacer nada».
- **WSL no debe apagarse** al cerrar la última terminal. Comprobación antes de la primera
  noche: lanzar una sesión de prueba (`tmux new-session -d -s prueba "sleep 600"`), cerrar
  todas las terminales de Ubuntu, esperar dos minutos, abrir una nueva y ver que `tmux ls`
  sigue mostrando `prueba`. Si no aparece, añadir en `.wslconfig`, sección `[wsl2]`,
  `vmIdleTimeout=-1`, ejecutar `wsl --shutdown` en PowerShell y repetir.

Los comandos concretos de cada fase se añaden aquí cuando existan.

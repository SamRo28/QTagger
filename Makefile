PYTHON_BOOTSTRAP ?= python3.13
CONFIG ?= configs/dataset.yaml
SPLITS ?= configs/splits.yaml
PROCESOS ?= 4
VENV ?= .venv
PYTHON := $(VENV)/bin/python
BIN := $(VENV)/bin

.PHONY: install lock lint test test-all dataset dataset-mini clean-data servidores parar-servidores estado-servidores ping

$(PYTHON):
	$(PYTHON_BOOTSTRAP) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip

## install: create the virtualenv and install qcclass (editable) pinned to requirements.lock
install: $(PYTHON)
	$(PYTHON) -m pip install -c requirements.lock -e ".[dev]"

## lock: regenerate requirements.lock from the current environment
lock: $(PYTHON)
	$(PYTHON) -m pip freeze --exclude-editable > requirements.lock

## lint: static checks
lint:
	$(BIN)/ruff check .

## test: fast test suite (skips tests marked as slow)
test:
	$(BIN)/pytest -m "not slow"

## test-all: full test suite, including slow integration tests
test-all:
	$(BIN)/pytest

## dataset: run the whole pipeline (override CONFIG, SPLITS, PROCESOS)
dataset:
	$(BIN)/qcclass todo --config $(CONFIG) --splits $(SPLITS) --procesos $(PROCESOS)

## dataset-mini: whole pipeline on the reduced configuration (writes data_mini/, results_mini/)
dataset-mini:
	$(MAKE) dataset CONFIG=configs/dataset_mini.yaml SPLITS=configs/splits_mini.yaml

## clean-data: delete every generated (git-ignored) file under data/, keeping data/README.md
clean-data:
	git clean -fdX -- data/

## servidores: start MLflow and the Optuna storage proxy (server machine; settings in .env)
servidores:
	$(BIN)/qcclass servidores arrancar

## parar-servidores: stop the servers started with `make servidores`
parar-servidores:
	$(BIN)/qcclass servidores parar

## estado-servidores: show whether the servers run and answer
estado-servidores:
	$(BIN)/qcclass servidores estado

## ping: check the connection with both servers (any machine; settings in .env)
ping:
	$(BIN)/qcclass ping

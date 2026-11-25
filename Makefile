PYTHON := $(if $(wildcard venv/bin/python),venv/bin/python,python3)
DATA ?= data
TIMEOUT ?= 1800
PLOT_SCRIPT := $(if $(wildcard src/plot_grafos.py),src/plot_grafos.py,src/plot.py)

.PHONY: help install validate features profiling bnb bnb-fast bnb-bg \
        gaps regression sensibilidade plots post all all-fast test web clean

help:
	@echo "Alvos disponíveis:"
	@echo "  make install        Instala dependências (requirements.txt)"
	@echo "  make validate       Valida leitura e simetria das instâncias TSPLIB"
	@echo "  make features       Extrai features -> results/features.csv"
	@echo "  make profiling      Profiling do Christofides/Twice-around + cProfile"
	@echo "  make bnb            Executa B&B oficial (TIMEOUT=$(TIMEOUT)s)"
	@echo "  make bnb-fast       Executa B&B rápido de teste (TIMEOUT=60s)"
	@echo "  make bnb-bg         Executa B&B oficial em segundo plano (nohup)"
	@echo "  make post           Roda gaps + regression + sensibilidade + plots"
	@echo "  make all            Pipeline oficial completo (validate..bnb..post)"
	@echo "  make all-fast       Pipeline completo com B&B de 60s"
	@echo "  make web            Inicia o visualizador Flask (app.py)"
	@echo "  make test           Executa testes em tests/"
	@echo "  make clean          Remove caches (__pycache__)"

install:
	$(PYTHON) -m pip install -r requirements.txt

validate:
	$(PYTHON) src/instancias.py $(DATA)

features:
	@mkdir -p results
	$(PYTHON) src/features.py $(DATA) results/features.csv

profiling:
	@mkdir -p results
	$(PYTHON) src/profilling_for_christofides.py $(DATA) 7 5

bnb:
	@mkdir -p results
	$(PYTHON) main.py bnb $(DATA) $(TIMEOUT)

bnb-fast:
	@mkdir -p results
	$(PYTHON) main.py bnb $(DATA) 60

bnb-bg:
	@mkdir -p results
	nohup $(PYTHON) -u main.py bnb $(DATA) $(TIMEOUT) > results/log_bnb.txt 2>&1 &
	@echo "B&B rodando em segundo plano. Acompanhe com: tail -f results/log_bnb.txt"

gaps:
	$(PYTHON) src/gaps.py

regression:
	$(PYTHON) src/regression.py

sensibilidade:
	$(PYTHON) src/sensibilidade.py

plots:
	$(PYTHON) $(PLOT_SCRIPT) ulysses16 ulysses22 bays29 berlin52

post: gaps regression sensibilidade plots

all: validate features profiling bnb post

all-fast: validate features profiling bnb-fast post

test:
	@for f in tests/*.py; do \
		[ "$$(basename "$$f")" = "__init__.py" ] && continue; \
		echo "=== Rodando $$f ==="; \
		$(PYTHON) "$$f" || exit 1; \
	done

web:
	$(PYTHON) app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
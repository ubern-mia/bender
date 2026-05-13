VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv
MODE ?= visualize

FILE ?= training-models/results/v1/model.onnx

.PHONY: setup install clean clean-venv netron run-v1 run-v2 run-v3 run-v4 run-v5 run-v6 run-v7

setup:
	$(UV) venv $(VENV)

install: setup
	$(UV) pip install --python $(PYTHON) -e .

netron:
	$(VENV)/bin/netron $(FILE)

clean:
	rm -rf training-models/results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-venv:
	rm -rf $(VENV)

run-v1:
	cd training-models && ../$(PYTHON) dermamnist_v1_initial.py $(MODE)

run-v2:
	cd training-models && ../$(PYTHON) dermamnist_v2_momentum0p9.py $(MODE)

run-v3:
	cd training-models && ../$(PYTHON) dermamnist_v3_lr0p005_val_patience.py $(MODE)

run-v4:
	cd training-models && ../$(PYTHON) dermamnist_v4_adam_TB.py $(MODE)

run-v5:
	cd training-models && ../$(PYTHON) dermamnist_v5_deeper_network.py $(MODE)

run-v6:
	cd training-models && ../$(PYTHON) dermamnist_v6_even_deeper_network.py $(MODE)

run-v7:
	cd training-models && ../$(PYTHON) dermamnist_v7_with_augm.py $(MODE)

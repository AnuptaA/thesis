VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# directories
DATASETS_DIR = datasets/synthetic/data
SIMULATIONS_DIR = simulations/synthetic/raw

.PHONY: all clean clobber help generate-datasets run-simulations analyze

all: help

help:
    @echo "unfinished"

generate-datasets:
    @echo "Generating synthetic datasets..."
    @mkdir -p $(DATASETS_DIR)
    $(PYTHON) datasets/synthetic/generator.py
    @echo "Done. Datasets saved to $(DATASETS_DIR)"

run-simulations:
    @echo "Running simulations..."
    @mkdir -p $(SIMULATIONS_DIR)
    $(PYTHON) simulations/synthetic/simulate.py
    @echo "Done. Results saved to $(SIMULATIONS_DIR)"

clean:
    @echo "Cleaning simulation results..."
    @rm -rf $(SIMULATIONS_DIR)
    @echo "Done."

clobber: clean
    @echo "Removing all generated data..."
    @rm -rf $(DATASETS_DIR)
    @rm -rf __pycache__ */__pycache__ */*/__pycache__
    @find . -name "*.pyc" -delete
    @echo "Done."
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all help
.PHONY: generate run analyze analyze-test
.PHONY: sift-prepare sift-run sift-analyze
.PHONY: esci-prepare esci-run esci-analyze
.PHONY: characterize characterize-synthetic characterize-sift characterize-esci
.PHONY: test test-distance-metrics test-kv-cache test-lemmas test-main-memory \
        test-perturbation test-verification test-simulate test-generator test-dataloaders \
        test-sift test-sift-edge-cases test-sift-e2e \
        test-esci test-esci-edge-cases test-esci-e2e
.PHONY: count-datasets
.PHONY: clean clean-synthetic clean-sift clean-esci clean-cache \
        clobber clobber-synthetic clobber-sift clobber-esci

all: help

help:
	@echo "Synthetic:  generate | run | analyze | analyze-test"
	@echo "SIFT:       sift-prepare | sift-run | sift-analyze"
	@echo "ESCI:       esci-prepare | esci-run | esci-analyze"
	@echo "Characterize: characterize | characterize-synthetic | characterize-sift | characterize-esci"
	@echo "Tests:      test | test-<name> | test-sift | test-esci"
	@echo "            names: distance-metrics kv-cache lemmas main-memory"
	@echo "                   perturbation verification simulate generator dataloaders"
	@echo "            sift:  test-sift-edge-cases test-sift-e2e"
	@echo "            esci:  test-esci-edge-cases test-esci-e2e"
	@echo "Utilities:  count-datasets"
	@echo "Cleanup:    clean[-synthetic|-sift|-esci] | clobber[-synthetic|-sift|-esci]"

# ── Synthetic workflow ────────────────────────────────────────────────────────

generate:
	$(PYTHON) datasets/synthetic/generator.py --test-set all

run:
	$(PYTHON) simulations/synthetic/run_synthetic.py

analyze:
	$(PYTHON) simulations/synthetic/analyze_synthetic.py
	$(PYTHON) simulations/synthetic/compile_pdf.py

analyze-test:
	$(PYTHON) tests/test_analysis.py

# ── SIFT workflow ─────────────────────────────────────────────────────────────

sift-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all

sift-run:
	$(PYTHON) simulations/sift/run_sift.py

sift-analyze:
	$(PYTHON) simulations/sift/analyze_sift.py

# ── ESCI workflow ─────────────────────────────────────────────────────────────

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

esci-run:
	$(PYTHON) simulations/esci/run_esci.py

esci-analyze:
	$(PYTHON) simulations/esci/analyze_esci.py

# ── Characterization ──────────────────────────────────────────────────────────

characterize-synthetic:
	$(PYTHON) simulations/characterize_queries.py --dataset synthetic \
	    --benchmarks set1_baseline_seed42 set1_baseline_seed43 set1_baseline_seed44

characterize-sift:
	$(PYTHON) simulations/characterize_queries.py --dataset sift \
	    --benchmarks sift_b50k_c1024k100_t512n20_s42 --full

characterize-esci:
	$(PYTHON) simulations/characterize_queries.py --dataset esci \
	    --benchmarks esci_c1024k99_t512n20_s42 --full

characterize: characterize-synthetic characterize-sift characterize-esci

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-distance-metrics test-kv-cache test-lemmas test-main-memory \
      test-perturbation test-verification test-simulate test-generator test-dataloaders \
      test-sift test-esci

test-distance-metrics:
	$(PYTHON) tests/util_tests/test_distance_metrics.py

test-kv-cache:
	$(PYTHON) tests/util_tests/test_kv_cache.py

test-lemmas:
	$(PYTHON) tests/util_tests/test_lemmas.py

test-main-memory:
	$(PYTHON) tests/util_tests/test_main_memory.py

test-perturbation:
	$(PYTHON) tests/util_tests/test_perturbation.py

test-verification:
	$(PYTHON) tests/util_tests/test_verification.py

test-simulate:
	$(PYTHON) tests/simulation_tests/test_simulate.py

test-generator:
	$(PYTHON) tests/generation_tests/test_generator.py

test-dataloaders:
	$(PYTHON) tests/dataloader_tests/test_dataloaders.py

test-sift: test-sift-edge-cases test-sift-e2e

test-sift-edge-cases:
	$(PYTHON) tests/sift_tests/test_edge_cases.py

test-sift-e2e:
	$(PYTHON) tests/sift_tests/test_e2e_mini.py

test-esci: test-esci-edge-cases test-esci-e2e

test-esci-edge-cases:
	$(PYTHON) tests/esci_tests/test_edge_cases.py

test-esci-e2e:
	$(PYTHON) tests/esci_tests/test_e2e_mini.py

# ── Utilities ─────────────────────────────────────────────────────────────────

count-datasets:
	@ls -1d datasets/synthetic/data/set*_* 2>/dev/null | wc -l

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: clean-synthetic clean-sift clean-esci

clean-synthetic:
	# rm -rf simulations/synthetic/raw/*
	# rm -rf simulations/synthetic/processed/*

clean-sift:
	# rm -rf simulations/sift/raw/*
	# rm -rf simulations/sift/processed/*

clean-esci:
	# rm -rf simulations/esci/raw/*
	# rm -rf simulations/esci/processed/*

clean-cache:
	find datasets/synthetic/data -name "ground_truth_cache" -type d -exec rm -rf {} + 2>/dev/null || true

clobber: clobber-synthetic clobber-sift clobber-esci
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clobber-synthetic: clean-synthetic clean-cache
	rm -rf datasets/synthetic/data/*

clobber-sift: clean-sift
	find datasets/sift -maxdepth 1 -type d -name "sift_b*" -exec rm -rf {} +

clobber-esci: clean-esci
	find datasets/esci -maxdepth 1 -type d -name "esci_c*" -exec rm -rf {} +

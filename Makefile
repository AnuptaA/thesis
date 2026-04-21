VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all help
.PHONY: synthetic-generate synthetic-run synthetic-analyze
.PHONY: sift-prepare sift-run sift-analyze
.PHONY: esci-prepare esci-run esci-analyze
.PHONY: characterize
.PHONY: test \
        test-utils test-distance-metrics test-kv-cache test-lemmas test-main-memory \
        test-perturbation test-verification test-simulate \
        test-synthetic test-generator test-e2e-synthetic test-analysis \
        test-sift test-sift-edge-cases test-sift-e2e test-dataloaders \
        test-esci test-esci-edge-cases test-esci-e2e
.PHONY: count-datasets
.PHONY: clobber clobber-synthetic clobber-sift clobber-esci clean-cache

all: help

help:
	@echo "Synthetic:  synthetic-generate | synthetic-run | synthetic-analyze"
	@echo "SIFT:       sift-prepare | sift-run | sift-analyze"
	@echo "ESCI:       esci-prepare | esci-run | esci-analyze"
	@echo "All:        analyze  (runs all workloads + collects results)"
	@echo "Characterize: characterize (all workloads, cross-workload plots)"
	@echo "Tests:      test | test-utils | test-synthetic | test-sift | test-esci"
	@echo "            utils:     test-distance-metrics test-kv-cache test-lemmas"
	@echo "                       test-main-memory test-perturbation test-verification test-simulate"
	@echo "            synthetic: test-generator test-e2e-synthetic test-analysis"
	@echo "            sift:      test-sift-edge-cases test-sift-e2e test-dataloaders"
	@echo "            esci:      test-esci-edge-cases test-esci-e2e"
	@echo "Utilities:  count-datasets"
	@echo "Cleanup:    clobber[-synthetic|-sift|-esci]"

# ── Synthetic workflow ────────────────────────────────────────────────────────

synthetic-generate:
	$(PYTHON) datasets/synthetic/generator.py --test-set all

synthetic-run:
	$(PYTHON) simulations/synthetic/run_synthetic.py

synthetic-analyze:
	$(PYTHON) simulations/synthetic/analyze_synthetic.py
	$(PYTHON) simulations/synthetic/compile_pdf.py

# ── SIFT workflow ─────────────────────────────────────────────────────────────

sift-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all

sift-run:
	$(PYTHON) simulations/sift/run_sift.py

sift-analyze:
	$(PYTHON) simulations/sift/analyze_sift.py
	$(PYTHON) simulations/sift/compile_pdf.py

# ── ESCI workflow ─────────────────────────────────────────────────────────────

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

esci-run:
	$(PYTHON) simulations/esci/run_esci.py

esci-analyze:
	$(PYTHON) simulations/esci/analyze_esci.py
	$(PYTHON) simulations/esci/compile_pdf.py

# ── Combined analysis ─────────────────────────────────────────────────────────

analyze: synthetic-analyze sift-analyze esci-analyze
	$(PYTHON) simulations/collect_results.py

# ── Characterization ──────────────────────────────────────────────────────────

characterize:
	$(PYTHON) simulations/characterize.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-utils test-synthetic test-sift test-esci

# utils tests
test-utils: test-distance-metrics test-kv-cache test-lemmas test-main-memory \
            test-perturbation test-verification test-simulate

test-distance-metrics:
	$(PYTHON) tests/utils/test_distance_metrics.py

test-kv-cache:
	$(PYTHON) tests/utils/test_kv_cache.py

test-lemmas:
	$(PYTHON) tests/utils/test_lemmas.py

test-main-memory:
	$(PYTHON) tests/utils/test_main_memory.py

test-perturbation:
	$(PYTHON) tests/utils/test_perturbation.py

test-verification:
	$(PYTHON) tests/utils/test_verification.py

test-simulate:
	$(PYTHON) tests/utils/test_simulate.py

# synthetic tests
test-synthetic: test-generator test-e2e-synthetic test-analysis

test-generator:
	$(PYTHON) tests/synthetic/test_generator.py

test-e2e-synthetic:
	$(PYTHON) tests/synthetic/test_e2e_mini.py

test-analysis:
	$(PYTHON) tests/synthetic/test_analysis.py

# sift tests
test-sift: test-sift-edge-cases test-sift-e2e test-dataloaders

test-sift-edge-cases:
	$(PYTHON) tests/sift/test_edge_cases.py

test-sift-e2e:
	$(PYTHON) tests/sift/test_e2e_mini.py

test-dataloaders:
	$(PYTHON) tests/sift/test_dataloaders.py

# esci tests
test-esci: test-esci-edge-cases test-esci-e2e

test-esci-edge-cases:
	$(PYTHON) tests/esci/test_edge_cases.py

test-esci-e2e:
	$(PYTHON) tests/esci/test_e2e_mini.py

# ── Utilities ─────────────────────────────────────────────────────────────────

count-datasets:
	@ls -1d datasets/synthetic/data/set*_* 2>/dev/null | wc -l

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-cache:
	find datasets/synthetic/data -name "ground_truth_cache" -type d -exec rm -rf {} + 2>/dev/null || true

clobber: clobber-synthetic clobber-sift clobber-esci
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clobber-synthetic: clean-cache
	rm -rf datasets/synthetic/data/*

clobber-sift:
	find datasets/sift -maxdepth 1 -type d -name "sift_b*" -exec rm -rf {} +

clobber-esci:
	find datasets/esci -maxdepth 1 -type d -name "esci_c*" -exec rm -rf {} +

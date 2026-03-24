#!/usr/bin/env python3
"""
End-to-end test for the ESCI simulation workflow using synthetic mini data.

No real ESCI data is required -- create_mini_esci_benchmark() generates
synthetic L2-normalized 384-dim vectors and a fake cache GT.

Steps tested:
  1. Create mini ESCI benchmark (synthetic vectors)
  2. Run run_benchmark() for all 6 lemma algorithms
  3. Verify output files: summary.json, per_query.csv, per-algo JSONs, logs
  4. Run analyze() and verify CSVs + summary JSON
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from simulations.esci.run_esci import run_benchmark
from simulations.esci.analyze_esci import analyze
from tests.esci_tests.helpers import create_mini_esci_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_e2e_workflow():
    """End-to-end test of complete ESCI workflow."""
    print("\n" + "="*80)
    print("Testing ESCI E2E Workflow")
    print("="*80)

    tmpdir = get_temp_dir("e2e_esci_")

    try:
        benchmark_dir = tmpdir / "mini_esci"
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"

        print("\n[1/3] Creating mini benchmark...")
        config = create_mini_esci_benchmark(benchmark_dir)
        print(f"  Cache queries: {config['num_cache_queries']} (K={config['cache_K']})")
        print(f"  Test queries:  {config['num_test_queries']} (N={config['test_N']})")
        print(f"  Dimension:     {config['dimension']}")

        print("\n[2/3] Running simulations...")
        algorithms = ['lemma1', 'lemma1_no_union', 'lemma2', 'lemma2_no_union',
                      'combined', 'combined_no_union']

        # run_benchmark expects the benchmark directory name and a raw_dir root
        # we need to place the benchmark under a datasets/esci/-like structure
        # for the test, we call it directly with the full path as benchmark_dir
        # and use a patched version via the output_dir arg
        results = run_benchmark(
            benchmark_name=benchmark_dir.name,
            algorithms=algorithms,
            output_dir=str(raw_dir),
            debug=False,
            benchmark_base_dir=str(tmpdir),
        )

        dataset_out = raw_dir / benchmark_dir.name
        assert dataset_out.exists(), f"Missing dataset output dir: {dataset_out}"

        summary_file = dataset_out / "summary.json"
        per_query_file = dataset_out / "per_query.csv"
        logs_dir = dataset_out / "logs"

        assert summary_file.exists(), "Missing summary.json"
        assert per_query_file.exists(), "Missing per_query.csv"
        assert logs_dir.exists(), "Missing logs directory"

        with open(summary_file) as f:
            summary = json.load(f)

        assert len(summary['results']) == len(algorithms), \
            f"Expected {len(algorithms)} results, got {len(summary['results'])}"

        lines = per_query_file.read_text().splitlines()
        assert len(lines) > 1, "per_query.csv is empty"

        print(f"  Generated {len(summary['results'])} algorithm results")
        print(f"  Generated {len(lines)-1} per-query rows")

        # verify at least hit_rate and metric fields are present
        for r in summary['results']:
            assert 'hit_rate' in r, f"Missing hit_rate in result: {r['algorithm']}"
            assert r['metric'] == 'angular', f"Expected angular metric, got {r['metric']}"

        print("\n[3/3] Running analysis...")
        analyze(str(raw_dir), str(processed_dir))

        assert processed_dir.exists(), "Missing processed directory"
        assert (processed_dir / "summary.csv").exists(),     "Missing summary.csv"
        assert (processed_dir / "summaries.json").exists(),  "Missing summaries.json"

        summary_lines = (processed_dir / "summary.csv").read_text().splitlines()
        assert len(summary_lines) > 1, "summary.csv is empty"

        print(f"  summary.csv:     {len(summary_lines)-1} rows")

        print("\n" + "="*80)
        print("Passed.")
        print("="*80)

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_e2e_debug_mode():
    """Test debug mode limits to 5 queries and still produces output files."""
    print("\n" + "="*80)
    print("Testing ESCI E2E Debug Mode")
    print("="*80)

    tmpdir = get_temp_dir("e2e_esci_debug_")

    try:
        benchmark_dir = tmpdir / "mini_esci_debug"
        raw_dir = tmpdir / "raw"

        create_mini_esci_benchmark(benchmark_dir)

        results = run_benchmark(
            benchmark_name=benchmark_dir.name,
            algorithms=['lemma1', 'combined'],
            output_dir=str(raw_dir),
            debug=True,
            benchmark_base_dir=str(tmpdir),
        )

        dataset_out = raw_dir / benchmark_dir.name
        assert (dataset_out / "summary.json").exists(), "Missing summary.json in debug mode"

        with open(dataset_out / "summary.json") as f:
            summary = json.load(f)

        # debug mode: 5 queries (mini has 20 test queries, debug limits to 5)
        for r in summary['results']:
            assert r['total_queries'] == 5, \
                f"Debug mode should limit to 5 queries, got {r['total_queries']}"

        print(f"Debug mode: {len(summary['results'])} algorithms, 5 queries each")
        print("\n" + "="*80)
        print("Passed.")
        print("="*80)

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def main():
    test_e2e_workflow()
    test_e2e_debug_mode()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

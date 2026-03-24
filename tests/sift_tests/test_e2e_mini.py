#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import json
from simulations.sift.run_sift import run_benchmark
from simulations.sift.analyze_sift import analyze
from tests.sift_tests.helpers import create_mini_sift_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_e2e_workflow():
    """
    End-to-end test of complete SIFT workflow.

    Steps:
    1. Create mini benchmark (with precomputed cache GT)
    2. Run simulations via run_benchmark
    3. Analyze results
    4. Verify outputs
    """
    print("\n" + "="*80)
    print("Testing SIFT E2E")
    print("="*80)

    tmpdir = get_temp_dir("e2e_workflow_")

    try:
        benchmark_name = "test_benchmark"
        benchmark_dir = tmpdir / benchmark_name
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"

        print("\n[1/3] Creating mini benchmark...")
        config = create_mini_sift_benchmark(benchmark_dir)

        print(f"Created: {config['num_base_vectors']} base vectors")
        print(f"Created: {config['num_cache_queries']} cache queries (K={config['cache_K']})")
        print(f"Created: {config['num_test_queries']} test queries (N={config['test_N']})")

        print("\n[2/3] Running simulations...")
        algorithms = ["lemma1", "lemma1_no_union", "lemma2", "lemma2_no_union",
                      "combined", "combined_no_union"]

        run_benchmark(
            benchmark_name=benchmark_name,
            algorithms=algorithms,
            output_dir=str(raw_dir),
            benchmark_base_dir=str(tmpdir),
        )

        dataset_dir = raw_dir / benchmark_name
        assert dataset_dir.exists(), "Missing dataset subdirectory"

        summary_file = dataset_dir / "summary.json"
        per_query_file = dataset_dir / "per_query.csv"
        logs_dir = dataset_dir / "logs"

        assert summary_file.exists(), "Missing summary.json"
        assert per_query_file.exists(), "Missing per_query.csv"
        assert logs_dir.exists(), "Missing logs directory"

        with open(summary_file) as f:
            summary = json.load(f)

        assert len(summary['results']) == len(algorithms), \
            f"Expected {len(algorithms)} results, got {len(summary['results'])}"

        lines = per_query_file.read_text().splitlines()
        assert len(lines) > 1, "per_query.csv is empty"

        print(f"Generated {len(summary['results'])} algorithm results")
        print(f"Generated {len(lines)-1} per-query rows")

        print("\n[3/3] Running analysis...")
        analyze(str(raw_dir), str(processed_dir))

        assert processed_dir.exists(), "Missing processed directory"

        summary_csv = processed_dir / "summary.csv"
        per_query_csv = processed_dir / "per_query_all.csv"
        summaries_json = processed_dir / "summaries.json"

        assert summary_csv.exists(), "Missing summary.csv"
        assert per_query_csv.exists(), "Missing per_query_all.csv"
        assert summaries_json.exists(), "Missing summaries.json"

        print(f"Generated summary.csv ({len(summary_csv.read_text().splitlines())-1} rows)")
        print(f"Generated per_query_all.csv")
        print(f"Generated summaries.json")

        print("\n" + "="*80)
        print("Passed.")
        print("="*80)

    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_e2e_debug_mode():
    """Test that debug=True limits to 5 queries per benchmark."""
    print("\n" + "="*80)
    print("Testing SIFT E2E debug mode")
    print("="*80)

    tmpdir = get_temp_dir("e2e_debug_")

    try:
        benchmark_name = "test_benchmark"
        benchmark_dir = tmpdir / benchmark_name
        raw_dir = tmpdir / "raw"

        create_mini_sift_benchmark(benchmark_dir)

        run_benchmark(
            benchmark_name=benchmark_name,
            algorithms=["lemma1"],
            output_dir=str(raw_dir),
            benchmark_base_dir=str(tmpdir),
            debug=True,
        )

        per_query_file = raw_dir / benchmark_name / "per_query.csv"
        assert per_query_file.exists(), "Missing per_query.csv"

        lines = per_query_file.read_text().splitlines()
        # header + up to 5 data rows for 1 algorithm
        assert len(lines) <= 6, f"Debug mode should limit to 5 queries, got {len(lines)-1}"

        print(f"Debug mode limited to {len(lines)-1} query rows")
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

#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from datasets.dataloaders import (
    load_sift_dataset,
    create_sift_benchmark_split,
    save_sift_benchmark
)
from utils.base import MainMemory

#-------------------------------------------------------------------------------

RESEARCH_BENCHMARKS = [
    # Set 1: Baseline (1024 cache x 3 seeds; shared with Set 2)
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=1024, num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    # Set 2: Cache size scaling (3 sizes x 3 seeds; 1024 shared with Set 1)
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=256,  num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=512,  num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=42, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=43, test_start=2048),
    dict(num_base_vectors=50_000, num_cache_queries=2048, num_test_queries=512,
         cache_K=100, test_N=20, seed=44, test_start=2048),
]

#-------------------------------------------------------------------------------

def create_benchmark_name(num_base: int, num_cache: int, num_test: int, cache_K: int, test_N: int, seed: int = 42) -> str:
    """
    Create unique benchmark directory name from parameters.

    Format: sift_b{base}_c{cache}k{K}_t{test}n{N}_s{seed}
    Example: sift_b50k_c1024k100_t512n20_s42
    """
    base_str = f"{num_base//1000}k" if num_base >= 1000 else str(num_base)
    if num_base >= 1_000_000:
        base_str = f"{num_base//1_000_000}m"

    return f"sift_b{base_str}_c{num_cache}k{cache_K}_t{num_test}n{test_N}_s{seed}"

#-------------------------------------------------------------------------------

def _compute_single_query_gt(args):
    """helper for parallel ground truth computation."""
    query_id, query, base_vectors, K, metric = args

    mm = MainMemory(vectors=list(base_vectors))
    top_k_vecs, top_k_dists, gap = mm.top_k_search(query, k=K, metric=metric)

    top_k_indices = []
    for vec in top_k_vecs:
        idx = np.where((base_vectors == vec).all(axis=1))[0][0]
        top_k_indices.append(idx)

    return query_id, np.array(top_k_indices), np.array(top_k_dists), gap

#-------------------------------------------------------------------------------

def _check_partial_gt(gt_file: Path, expected_count: int) -> tuple[dict, set]:
    """load partial ground truth from disk if it exists."""
    if not gt_file.exists():
        return {}, set()

    try:
        data = np.load(gt_file)
        ground_truth = {}
        completed = set()

        for i in range(expected_count):
            if f'indices_{i}' in data:
                ground_truth[i] = (
                    data[f'indices_{i}'],
                    data[f'distances_{i}'],
                    float(data[f'gap_{i}'])
                )
                completed.add(i)

        if completed:
            print(f"  Found partial ground truth: {len(completed)}/{expected_count} completed")

        return ground_truth, completed
    except Exception as e:
        print(f"  Warning: could not load partial ground truth: {e}")
        return {}, set()

#-------------------------------------------------------------------------------

def precompute_cache_ground_truth(
    base_vectors: np.ndarray,
    cache_queries: np.ndarray,
    K: int,
    output_dir: Path,
    num_workers: int = None,
    checkpoint_interval: int = 100
) -> dict:
    """
    Precompute top-K ground truth for cache queries using angular metric.
    Results are saved to output_dir/cache_gt_angular_K{K}.npz.
    Supports resuming from partial completion.

    Args:
        base_vectors: base vector array (M x D)
        cache_queries: cache query vectors (C x D)
        K: number of neighbors per cache entry
        output_dir: benchmark directory to save ground truth
        num_workers: parallel workers (default: cpu_count - 1)
        checkpoint_interval: save to disk every N queries

    Returns:
        dict mapping query_id -> (indices, distances, gap)
    """
    gt_file = output_dir / f"cache_gt_angular_K{K}.npz"
    ground_truth, completed = _check_partial_gt(gt_file, len(cache_queries))

    if len(completed) == len(cache_queries):
        print(f"Cache ground truth already complete ({len(cache_queries)} queries).")
        return ground_truth

    remaining = len(cache_queries) - len(completed)
    print(f"\nPrecomputing cache ground truth: {remaining}/{len(cache_queries)} remaining...")
    print(f"  Metric: angular, K: {K}")

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"  Using {num_workers} parallel workers")
    print(f"  Checkpointing every {checkpoint_interval} queries")

    args_list = [
        (i, cache_queries[i], base_vectors, K, "angular")
        for i in range(len(cache_queries))
        if i not in completed
    ]

    with Pool(num_workers) as pool:
        for chunk_start in range(0, len(args_list), checkpoint_interval):
            chunk_end = min(chunk_start + checkpoint_interval, len(args_list))
            chunk = args_list[chunk_start:chunk_end]

            print(f"\n  Processing chunk {chunk_start // checkpoint_interval + 1}/"
                  f"{(len(args_list) + checkpoint_interval - 1) // checkpoint_interval}...")

            results = list(tqdm(
                pool.imap(_compute_single_query_gt, chunk),
                total=len(chunk),
                desc="  Computing cache GT",
                file=sys.stderr
            ))

            for query_id, indices, distances, gap in results:
                ground_truth[query_id] = (indices, distances, gap)

            print(f"  Saving checkpoint... ({len(ground_truth)}/{len(cache_queries)} complete)")
            save_dict = {}
            for i, (indices, distances, gap) in ground_truth.items():
                save_dict[f'indices_{i}'] = indices
                save_dict[f'distances_{i}'] = distances
                save_dict[f'gap_{i}'] = gap
            np.savez_compressed(gt_file, **save_dict)

    print(f"\nCache ground truth complete: {gt_file}")
    return ground_truth

#-------------------------------------------------------------------------------

def prepare_sift_benchmark(
    num_base_vectors: int = 1_000_000,
    num_cache_queries: int = 9000,
    num_test_queries: int = 1000,
    cache_K: int = 100,
    test_N: int = 10,
    seed: int = 42,
    force: bool = False,
    num_workers: int = None,
    test_start: int = None,
):
    """
    Prepare SIFT benchmark with specified parameters.

    Computes and saves cache query ground truth (angular metric) to the benchmark
    directory so run_sift.py can load it directly without any brute-force computation.

    Args:
        num_base_vectors: number of base vectors (max 1M from SIFT1M)
        num_cache_queries: number of cache queries
        num_test_queries: number of test queries
        cache_K: K for cache queries (top-K)
        test_N: N for test queries (top-N)
        seed: random seed for reproducibility
        force: overwrite existing benchmark if True
        num_workers: parallel workers for ground truth computation
    """

    # load full SIFT dataset
    print("Loading SIFT dataset...")
    base, _, query, _ = load_sift_dataset("datasets/sift")

    # validate parameters
    if num_base_vectors > len(base):
        print(f"Warning: requested {num_base_vectors:,} base vectors but SIFT only has {len(base):,}")
        print(f"Using all {len(base):,} vectors")
        num_base_vectors = len(base)

    if num_cache_queries + num_test_queries > len(query):
        raise ValueError(
            f"Not enough queries: requested {num_cache_queries + num_test_queries}, "
            f"available {len(query)}"
        )

    # create benchmark name
    benchmark_name = create_benchmark_name(
        num_base_vectors, num_cache_queries, num_test_queries, cache_K, test_N, seed
    )

    output_dir = Path("datasets/sift") / benchmark_name

    # check if benchmark split already exists
    gt_file = output_dir / f"cache_gt_angular_K{cache_K}.npz"
    if output_dir.exists() and gt_file.exists() and not force:
        print(f"\n{'='*80}")
        print(f"Benchmark already exists: {output_dir}")
        print(f"{'='*80}")
        return str(output_dir)

    print(f"\n{'='*80}")
    print("Preparing SIFT benchmark with the following configuration:")
    print(f"{'='*80}")
    print(f"\nConfiguration:")
    print(f"  Base vectors:     {num_base_vectors:>10,}")
    print(f"  Cache queries:    {num_cache_queries:>10,} (K={cache_K})")
    print(f"  Test queries:     {num_test_queries:>10,} (N={test_N})")
    print(f"  Dimension:        {base.shape[1]:>10,}")
    print(f"  Seed:             {seed:>10}")
    print(f"\nOutput: {output_dir}")

    # subsample base vectors if needed
    if num_base_vectors < len(base):
        print(f"\nSubsampling base vectors: {num_base_vectors:,} / {len(base):,}")
        base_subset = base[:num_base_vectors]
    else:
        base_subset = base

    # create config
    config = {
        "dataset": "SIFT1M",
        "benchmark_name": benchmark_name,
        "num_base_vectors": num_base_vectors,
        "num_cache_queries": num_cache_queries,
        "num_test_queries": num_test_queries,
        "cache_K": cache_K,
        "test_N": test_N,
        "dimension": int(base.shape[1]),
        "seed": seed,
        "description": (
            f"SIFT benchmark: {num_base_vectors:,} base, "
            f"{num_cache_queries:,} cache (K={cache_K}), "
            f"{num_test_queries:,} test (N={test_N})"
        )
    }

    # create benchmark split
    print("\nCreating benchmark split...")
    if test_start is not None:
        print(f"  test_start={test_start} (fixed test queries across cache sizes)")
    base_out, cache_q, test_q = create_sift_benchmark_split(
        base_subset, query,
        num_cache_queries=num_cache_queries,
        num_test_queries=num_test_queries,
        cache_K=cache_K,
        test_N=test_N,
        seed=seed,
        test_start=test_start,
    )

    # save benchmark vectors and config
    if not output_dir.exists() or force:
        print(f"\nSaving benchmark to {output_dir}...")
        save_sift_benchmark(str(output_dir), base_out, cache_q, test_q, config)

    # compute and save cache query ground truth (angular, needed by run_sift.py)
    print("\n[GT] Computing cache query ground truth (angular metric)...")
    precompute_cache_ground_truth(
        base_vectors=base_out,
        cache_queries=cache_q,
        K=cache_K,
        output_dir=output_dir,
        num_workers=num_workers
    )

    print(f"\n{'='*80}")
    print("Benchmark prepared.")
    print(f"{'='*80}")
    print(f"Location: {output_dir}")

    return str(output_dir)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare SIFT benchmark with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--all',
        action='store_true',
        help='Prepare all research benchmarks'
    )
    group.add_argument(
        '--preset',
        type=str,
        choices=['tiny'],
        help='Use preset configuration'
    )

    parser.add_argument('--base', type=int, help='Number of base vectors')
    parser.add_argument('--cache', type=int, help='Number of cache queries')
    parser.add_argument('--test', type=int, help='Number of test queries')
    parser.add_argument('--cache-k', type=int, help='K for cache queries')
    parser.add_argument('--test-n', type=int, default=20, help='N for test queries')
    parser.add_argument('--test-start', type=int, default=None,
                        help='Fixed start index for test queries (Set 2 cache scaling)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Overwrite existing benchmark')
    parser.add_argument('--workers', type=int, default=None, help='Parallel workers for GT computation')

    args = parser.parse_args()

    presets = {
        'tiny': {
            'num_base_vectors': 10_000,
            'num_cache_queries': 100,
            'cache_K': 20,
            'num_test_queries': 50,
            'test_N': 10,
            'description': 'Tiny test (10K base, 100 cache, 50 test)'
        }
    }

    if args.all:
        print(f"\n{'='*80}")
        print(f"Preparing all {len(RESEARCH_BENCHMARKS)} SIFT research benchmarks")
        print(f"{'='*80}\n")
        failed = []
        for i, bench_params in enumerate(RESEARCH_BENCHMARKS):
            print(f"\n[{i+1}/{len(RESEARCH_BENCHMARKS)}] {bench_params}")
            try:
                p = dict(bench_params)
                p['force'] = args.force
                p['num_workers'] = args.workers
                prepare_sift_benchmark(**p)
            except Exception as e:
                print(f"ERROR: {e}")
                failed.append(bench_params)
        if failed:
            print(f"\nFailed: {len(failed)} benchmarks")
            sys.exit(1)
    elif args.preset:
        params = presets[args.preset].copy()
        print(f"Using preset '{args.preset}': {params.pop('description')}")
        params['seed'] = args.seed
        params['force'] = args.force
        params['num_workers'] = args.workers
        prepare_sift_benchmark(**params)
    else:
        params = {
            'num_base_vectors': args.base or 50_000,
            'num_cache_queries': args.cache or 1024,
            'num_test_queries': args.test or 512,
            'cache_K': args.cache_k or 100,
            'test_N': args.test_n,
            'seed': args.seed,
            'force': args.force,
            'num_workers': args.workers,
            'test_start': args.test_start,
        }
        prepare_sift_benchmark(**params)

#-------------------------------------------------------------------------------

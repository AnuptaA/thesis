#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import argparse
from datasets.dataloaders import load_esci_data, save_esci_benchmark

#-------------------------------------------------------------------------------

ESCI_DIR = Path("datasets/esci")

RESEARCH_BENCHMARKS = [
    # Set 1: Baseline -- 1024 cache x 3 seeds
    # test_start=5000 matches Set 2 so 1024 shared with Set 2 uses identical test queries
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=1024, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # Set 2: Cache size scaling -- 5 sizes x 3 seeds
    # test_start=5000 anchors test queries beyond max cache size (4096) so all sizes use
    # identical test queries (perm[5000:5512]). 1024 is shared with Set 1.
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=256,  num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=512,  num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # 1024 already covered in Set 1
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=2048, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=42, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=43, test_start=5000),
    dict(num_cache=4096, num_test=512, cache_K=99, test_N=20, seed=44, test_start=5000),
    # Set 3: K/N relationship -- 9 pairs x 3 seeds; (K,N) in {20,50,99}^2
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=20,  test_N=99,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=50,  test_N=99,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=20,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=50,  seed=44),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=42),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=43),
    dict(num_cache=1024, num_test=256, cache_K=99,  test_N=99,  seed=44),
]

#-------------------------------------------------------------------------------

def create_benchmark_name(num_cache: int, cache_K: int, num_test: int, test_N: int, seed: int) -> str:
    """
    Create benchmark directory name.
    Format: esci_c{cache}k{K}_t{test}n{N}_s{seed}
    """
    return f"esci_c{num_cache}k{cache_K}_t{num_test}n{test_N}_s{seed}"

#-------------------------------------------------------------------------------

def build_cache_gt(
    cache_query_indices: np.ndarray,
    topk_ids: np.ndarray,
    topk_scores: np.ndarray,
    product_embeddings: np.ndarray,
    cache_K: int,
    output_dir: Path
):
    """
    Build and save cache query ground truth from precomputed topk data.

    Converts cosine similarity scores to angular distances and saves the
    top-K product vectors (needed for KVCache population at simulation time).

    Args:
        cache_query_indices: indices into the full query set (130K) for cache queries
        topk_ids: precomputed top-100 product indices, shape (num_queries, 100)
        topk_scores: precomputed top-100 cosine scores, shape (num_queries, 100)
        product_embeddings: full product embedding array, shape (1.82M, 384)
        cache_K: number of neighbors to use (must be <= 100)
        output_dir: benchmark directory to save cache_gt_K{K}.npz
    """
    assert cache_K < 100, f"cache_K must be < 100 to derive gap from precomputed top-100 shards (got {cache_K}); use cache_K=99 instead of 100"

    gt_file = output_dir / f"cache_gt_K{cache_K}.npz"
    if gt_file.exists():
        print(f"  Cache GT already exists: {gt_file.name}")
        return

    print(f"  Building cache GT for {len(cache_query_indices)} cache queries (K={cache_K})...")

    save_dict = {}
    for local_i, global_i in enumerate(cache_query_indices):
        # fetch K+1 entries from precomputed top-100 to compute true gap D(Q,e_{K+1}) - D(Q,e_K)
        # cache_K < 100 is asserted above, so index cache_K is always in bounds
        ids = topk_ids[global_i, :cache_K]               # (K,) product indices
        scores_kp1 = topk_scores[global_i, :cache_K + 1] # (K+1,) cosine similarity

        # convert cosine similarity -> angular distance, normalized by pi -> [0, 1]
        # matches angular_distance() in utils/base/distance_metrics.py
        distances_kp1 = (np.arccos(np.clip(scores_kp1, -1.0, 1.0)) / np.pi).astype(np.float32)
        distances = distances_kp1[:cache_K]              # (K,) distances for cache entry

        # true gap: D(Q, e_{K+1}) - D(Q, e_K)
        gap = float(distances_kp1[cache_K] - distances_kp1[cache_K - 1])

        # actual product vectors needed for KVCache
        vectors = product_embeddings[ids]  # (K, D)

        save_dict[f'indices_{local_i}'] = ids
        save_dict[f'vectors_{local_i}'] = vectors
        save_dict[f'distances_{local_i}'] = distances
        save_dict[f'gap_{local_i}'] = np.float32(gap)

    np.savez_compressed(gt_file, **save_dict)
    print(f"  Saved cache GT: {gt_file.name}")

#-------------------------------------------------------------------------------

def prepare_esci_benchmark(
    num_cache_queries: int = 1024,
    num_test_queries: int = 512,
    cache_K: int = 100,
    test_N: int = 20,
    seed: int = 42,
    force: bool = False,
    test_start: int = None,
):
    """
    Prepare a single ESCI benchmark.

    Samples (num_cache_queries + num_test_queries) queries disjointly from the
    full 130K query set, then slices top-K GT from topk_100shards.npz.
    Saves: cache_queries.npy, test_queries.npy, cache_gt_K{K}.npz, config.json.

    Args:
        num_cache_queries: number of cache population queries
        num_test_queries: number of test queries
        cache_K: K for cache entries (top-K neighbors); must be <= 100
        test_N: N for test queries (top-N to retrieve at query time)
        seed: random seed
        force: overwrite existing benchmark
    """
    benchmark_name = create_benchmark_name(num_cache_queries, cache_K, num_test_queries, test_N, seed)
    output_dir = ESCI_DIR / benchmark_name
    gt_file = output_dir / f"cache_gt_K{cache_K}.npz"

    if output_dir.exists() and gt_file.exists() and not force:
        print(f"Benchmark already exists: {benchmark_name}")
        return str(output_dir)

    print(f"\n{'='*80}")
    print(f"Preparing ESCI benchmark: {benchmark_name}")
    print(f"{'='*80}")
    print(f"  Cache queries: {num_cache_queries:,} (K={cache_K})")
    print(f"  Test queries:  {num_test_queries:,} (N={test_N})")
    print(f"  Seed:          {seed}")

    total_queries_needed = num_cache_queries + num_test_queries
    print(f"\nLoading ESCI data (need {total_queries_needed:,} of 130,652 queries)...")

    # Load embeddings -- needed once to extract top-K vectors for cache GT
    product_embeddings, query_vectors, topk_ids, topk_scores = load_esci_data(
        str(ESCI_DIR), load_embeddings=True
    )

    if total_queries_needed > len(query_vectors):
        raise ValueError(
            f"Not enough queries: need {total_queries_needed}, have {len(query_vectors)}"
        )

    # sample disjoint cache + test query indices
    # test_start fixes the anchor for test queries so all cache-size variants
    # share identical test queries (important for Set 2 cache-scaling comparison)
    if test_start is None:
        test_start = num_cache_queries
    np.random.seed(seed)
    perm = np.random.permutation(len(query_vectors))
    cache_indices = perm[:num_cache_queries]
    test_indices = perm[test_start:test_start + num_test_queries]
    if test_start < num_cache_queries:
        raise ValueError(
            f"test_start={test_start} < num_cache_queries={num_cache_queries}: "
            "test queries would overlap with cache queries"
        )

    cache_queries = query_vectors[cache_indices]
    test_queries = query_vectors[test_indices]

    config = {
        "dataset": "ESCI",
        "benchmark_name": benchmark_name,
        "num_cache_queries": num_cache_queries,
        "num_test_queries": num_test_queries,
        "cache_K": cache_K,
        "test_N": test_N,
        "dimension": int(query_vectors.shape[1]),
        "num_base_vectors": int(product_embeddings.shape[0]),
        "seed": seed,
        "description": (
            f"ESCI benchmark: full 1.82M base, "
            f"{num_cache_queries} cache (K={cache_K}), "
            f"{num_test_queries} test (N={test_N}), seed={seed}"
        ),
    }

    # save benchmark split
    save_esci_benchmark(str(output_dir), cache_queries, test_queries, config)

    # build and save cache GT (sliced from topk_100shards.npz)
    build_cache_gt(
        cache_query_indices=cache_indices,
        topk_ids=topk_ids,
        topk_scores=topk_scores,
        product_embeddings=product_embeddings,
        cache_K=cache_K,
        output_dir=output_dir,
    )

    print(f"\nBenchmark ready: {output_dir}")
    return str(output_dir)

#-------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare ESCI benchmarks (full 1.82M base, GT from topk_100shards.npz)",
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
        choices=['baseline', 'cache_scaling', 'kn_relationship'],
        help='Prepare benchmarks for one test set'
    )

    parser.add_argument('--cache', type=int, default=1024, help='Number of cache queries')
    parser.add_argument('--test', type=int, default=512, help='Number of test queries')
    parser.add_argument('--cache-k', type=int, default=100, help='K for cache entries (max 100)')
    parser.add_argument('--test-n', type=int, default=20, help='N for test queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Overwrite existing benchmarks')

    args = parser.parse_args()

    # define preset subsets
    preset_configs = {
        'baseline': [b for b in RESEARCH_BENCHMARKS
                     if b['num_cache'] in (512, 1024, 2048) and b['cache_K'] == 100 and b['test_N'] == 20],
        'cache_scaling': [b for b in RESEARCH_BENCHMARKS
                          if b['cache_K'] == 100 and b['test_N'] == 20],
        'kn_relationship': [b for b in RESEARCH_BENCHMARKS
                            if b['num_cache'] == 1024 and b['num_test'] == 256],
    }

    if args.all:
        configs = RESEARCH_BENCHMARKS
    elif args.preset:
        configs = preset_configs[args.preset]
    else:
        # single benchmark from CLI args
        configs = [dict(
            num_cache=args.cache,
            num_test=args.test,
            cache_K=args.cache_k,
            test_N=args.test_n,
            seed=args.seed,
        )]

    print(f"\n{'='*80}")
    print(f"ESCI Benchmark Preparation")
    print(f"{'='*80}")
    print(f"Benchmarks to prepare: {len(configs)}")
    print(f"{'='*80}\n")

    failed = []
    for cfg in configs:
        try:
            prepare_esci_benchmark(
                num_cache_queries=cfg['num_cache'],
                num_test_queries=cfg['num_test'],
                cache_K=cfg['cache_K'],
                test_N=cfg['test_N'],
                seed=cfg['seed'],
                force=args.force,
                test_start=cfg.get('test_start'),
            )
        except Exception as e:
            name = create_benchmark_name(cfg['num_cache'], cfg['cache_K'], cfg['num_test'], cfg['test_N'], cfg['seed'])
            print(f"\nERROR: {name} failed: {e}")
            failed.append(name)

    print(f"\n{'='*80}")
    print(f"Done: {len(configs) - len(failed)}/{len(configs)} benchmarks succeeded")
    if failed:
        print("Failed:")
        for b in failed:
            print(f"  {b}")
    print(f"{'='*80}")

    import sys
    sys.exit(1 if failed else 0)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#-------------------------------------------------------------------------------

#!/usr/bin/env python3
"""
Helper functions for ESCI tests.

Creates synthetic mini benchmarks using random L2-normalized vectors (384-dim)
so tests run without any real ESCI data files.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import tempfile
import shutil

#-------------------------------------------------------------------------------

def create_mini_esci_benchmark(benchmark_dir: Path, seed: int = 42):
    """
    Create minimal ESCI-like benchmark for testing.

    Uses synthetic random L2-normalized 384-dim vectors.
    Saves: cache_queries.npy, test_queries.npy, cache_gt_K{K}.npz, config.json.

    The cache GT contains synthetic top-K neighbors drawn randomly from
    a small synthetic product set -- enough to test that the simulation
    pipeline runs end-to-end without real ESCI data.

    Args:
        benchmark_dir: directory to save benchmark files
        seed: random seed for reproducibility

    Returns:
        config dict
    """
    np.random.seed(seed)

    dimension = 384
    num_cache = 20
    num_test = 20
    K = 10
    N = 5
    num_products = 200  # synthetic product pool for GT

    # synthetic L2-normalized vectors
    def rand_unit(n, d):
        v = np.random.randn(n, d).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    product_pool = rand_unit(num_products, dimension)
    cache_queries = rand_unit(num_cache, dimension)
    test_queries = rand_unit(num_test, dimension)

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / "cache_queries.npy", cache_queries)
    np.save(benchmark_dir / "test_queries.npy", test_queries)

    # build synthetic cache GT
    # for each cache query, pick K random products (with fake distances/gap)
    save_dict = {}
    for i in range(num_cache):
        # random top-K product indices (no duplicates)
        indices = np.random.choice(num_products, size=K, replace=False).astype(np.int64)
        vectors = product_pool[indices]  # (K, D)
        # fake angular distances in [0, 1] (normalized by pi, same as angular_distance()),
        # kept below 0.8 so gap/2 < 0.5 and lemma conditions don't trivially fire
        distances = np.sort(np.random.uniform(0.3, 0.8, size=K).astype(np.float32))
        gap = float(distances[-1])

        save_dict[f'indices_{i}'] = indices
        save_dict[f'vectors_{i}'] = vectors
        save_dict[f'distances_{i}'] = distances
        save_dict[f'gap_{i}'] = np.float32(gap)

    np.savez_compressed(benchmark_dir / f"cache_gt_K{K}.npz", **save_dict)

    config = {
        'dataset': 'ESCI',
        'benchmark_name': benchmark_dir.name,
        'num_cache_queries': num_cache,
        'num_test_queries': num_test,
        'cache_K': K,
        'test_N': N,
        'dimension': dimension,
        'num_base_vectors': 1814924,  # full catalog size (not loaded in tests)
        'seed': seed,
        'description': (
            f'Mini ESCI test benchmark: {num_cache} cache (K={K}), '
            f'{num_test} test (N={N}), 384-dim synthetic vectors'
        ),
    }

    with open(benchmark_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def get_temp_dir(prefix: str = "esci_test_") -> Path:
    """Create and return a temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))

#-------------------------------------------------------------------------------

def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

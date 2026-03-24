#!/usr/bin/env python3
"""
Helper functions for SIFT tests.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
import tempfile
import shutil
from utils.base import MainMemory

#-------------------------------------------------------------------------------

def create_mini_sift_benchmark(benchmark_dir: Path, seed=42):
    """
    Create minimal SIFT-like benchmark for testing.
    
    Args:
        benchmark_dir: directory to save benchmark files
        seed: random seed for reproducibility
        
    Returns:
        config dict with benchmark parameters
    """
    np.random.seed(seed)
    
    dimension = 128
    num_base = 100
    num_cache = 20
    num_test = 10
    K = 10
    N = 5
    
    base_vectors = np.random.randn(num_base, dimension).astype(np.float32)
    base_vectors /= np.linalg.norm(base_vectors, axis=1, keepdims=True)
    
    cache_queries = np.random.randn(num_cache, dimension).astype(np.float32)
    cache_queries /= np.linalg.norm(cache_queries, axis=1, keepdims=True)
    
    test_queries = np.random.randn(num_test, dimension).astype(np.float32)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)
    
    
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    np.save(benchmark_dir / "base_vectors.npy", base_vectors)
    np.save(benchmark_dir / "cache_queries.npy", cache_queries)
    np.save(benchmark_dir / "test_queries.npy", test_queries)

    # precompute angular cache ground truth (mirrors prepare_benchmark.py)
    mm = MainMemory(vectors=list(base_vectors))
    gt_data = {}
    for i, q in enumerate(cache_queries):
        top_k_vecs, top_k_dists, gap = mm.top_k_search(q, k=K, metric='angular')
        indices = []
        for vec in top_k_vecs:
            idx = int(np.where((base_vectors == vec).all(axis=1))[0][0])
            indices.append(idx)
        gt_data[i] = (np.array(indices), np.array(top_k_dists), gap)

    save_dict = {}
    for i, (indices, distances, gap) in gt_data.items():
        save_dict[f'indices_{i}'] = indices
        save_dict[f'distances_{i}'] = distances
        save_dict[f'gap_{i}'] = np.float32(gap)
    np.savez(benchmark_dir / f'cache_gt_angular_K{K}.npz', **save_dict)

    config = {
        'dataset': 'test_mini',
        'num_base_vectors': num_base,
        'num_cache_queries': num_cache,
        'num_test_queries': num_test,
        'cache_K': K,
        'test_N': N,
        'dimension': dimension,
        'seed': seed,
        'description': f'Mini test benchmark: {num_base} base, {num_cache} cache (K={K}), {num_test} test (N={N})',
    }

    with open(benchmark_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    return config

#-------------------------------------------------------------------------------

def get_temp_dir(prefix="sift_test_"):
    """Create and return temporary directory."""
    return Path(tempfile.mkdtemp(prefix=prefix))

#-------------------------------------------------------------------------------

def cleanup_temp_dir(temp_dir: Path):
    """Clean up temporary directory."""
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

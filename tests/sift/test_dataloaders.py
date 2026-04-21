#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import shutil
from datasets.dataloaders import (
    load_fvecs,
    load_ivecs,
    load_sift_dataset,
    create_sift_benchmark_split,
    save_sift_benchmark,
    load_sift_benchmark
)

#-------------------------------------------------------------------------------

def test_load_fvecs():
    """Test loading fvecs files."""
    print("Test 1: Load fvecs format")
    
    sift_path = Path("datasets/sift")
    if not (sift_path / "sift_query.fvecs").exists():
        assert True, "SIFT files not found, skipping test"
        return
    
    queries = load_fvecs(str(sift_path / "sift_query.fvecs"))
    
    assert queries.shape == (10000, 128), f"Expected (10000, 128), got {queries.shape}"
    assert queries.dtype == np.float32
    
    print(f"Loaded queries: {queries.shape}")

#-------------------------------------------------------------------------------

def test_load_ivecs():
    """Test loading ivecs files."""
    print("\nTest 2: Load ivecs format")
    
    sift_path = Path("datasets/sift")
    if not (sift_path / "sift_groundtruth.ivecs").exists():
        assert True, "SIFT files not found, skipping test"
        return
    
    groundtruth = load_ivecs(str(sift_path / "sift_groundtruth.ivecs"))
    
    assert groundtruth.shape == (10000, 100), f"Expected (10000, 100), got {groundtruth.shape}"
    assert groundtruth.dtype == np.int32
    
    print(f"Loaded ground truth: {groundtruth.shape}")

#-------------------------------------------------------------------------------

def test_load_sift_dataset():
    """Test loading complete SIFT dataset."""
    print("\nTest 3: Load complete SIFT dataset")
    
    sift_path = Path("datasets/sift")
    if not sift_path.exists():
        assert True, "SIFT directory not found, skipping test"
        return
    
    try:
        base, learn, query, gt = load_sift_dataset(str(sift_path))
        
        assert base.shape == (1000000, 128), f"Base shape wrong: {base.shape}"
        assert learn.shape == (100000, 128), f"Learn shape wrong: {learn.shape}"
        assert query.shape == (10000, 128), f"Query shape wrong: {query.shape}"
        assert gt.shape == (10000, 100), f"GT shape wrong: {gt.shape}"
        
        print("All SIFT files loaded correctly")
    except FileNotFoundError as e:
        print(f"Missing file: {e}")

#-------------------------------------------------------------------------------

def test_benchmark_split():
    """Test creating benchmark split."""
    print("\nTest 4: Create benchmark split")
    
    base = np.random.randn(1000, 128).astype(np.float32)
    queries = np.random.randn(10000, 128).astype(np.float32)
    
    base_out, cache_q, test_q = create_sift_benchmark_split(
        base, queries,
        num_cache_queries=9000,
        num_test_queries=1000,
        cache_K=100,
        test_N=10,
        seed=42
    )

    assert base_out.shape == (1000, 128)
    assert cache_q.shape == (9000, 128)
    assert test_q.shape == (1000, 128)
    
    print("Benchmark split created correctly")

#-------------------------------------------------------------------------------

def test_save_load_benchmark():
    """Test saving and loading benchmark."""
    print("\nTest 5: Save and load benchmark")
    
    base = np.random.randn(100, 128).astype(np.float32)
    cache_q = np.random.randn(90, 128).astype(np.float32)
    test_q = np.random.randn(10, 128).astype(np.float32)

    config = {
        "num_cache_queries": 90,
        "num_test_queries": 10,
        "cache_K": 100,
        "test_N": 10,
        "seed": 42
    }

    tmp_dir = "tests/dataloader_tests/tmp_benchmark"

    save_sift_benchmark(tmp_dir, base, cache_q, test_q, config)

    base_l, cache_l, test_l, config_l = load_sift_benchmark(tmp_dir)

    assert np.array_equal(base, base_l)
    assert np.array_equal(cache_q, cache_l)
    assert np.array_equal(test_q, test_l)
    assert config == config_l
    
    # cleanup
    shutil.rmtree(tmp_dir)

    print("Save/load works correctly.")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing SIFT Dataloaders")
    print("="*80)
    
    test_load_fvecs()
    test_load_ivecs()
    test_load_sift_dataset()
    test_benchmark_split()
    test_save_load_benchmark()
    
    print("\n" + "="*80)
    print("All tests passed.")
    print("="*80)

#-------------------------------------------------------------------------------
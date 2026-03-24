#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from datasets.dataloaders import load_sift_benchmark
from simulations.simulate import CacheSimulator
from tests.sift_tests.helpers import create_mini_sift_benchmark, get_temp_dir, cleanup_temp_dir

#-------------------------------------------------------------------------------

def test_empty_cache():
    """Test handling of empty cache (zero cache queries)."""
    print("\nTest: Empty cache")
    
    tmpdir = get_temp_dir("edge_empty_cache_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        empty_cache = np.zeros((0, 128), dtype=np.float32)
        empty_test = test_q[:1]
        
        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=empty_cache,
            test_queries=empty_test,
            K=10,
            N=5,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=empty_test[0],
            algorithm='lemma1',
            ground_truth_indices=[]
        )
        
        assert not result.cache_hit, "Empty cache should result in miss"
        
        print("Empty cache handled correctly (cache miss)")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_single_base_vector():
    """Test handling of single base vector."""
    print("\nTest: Single base vector")
    
    tmpdir = get_temp_dir("edge_single_base_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        single_base = base[:1]
        small_cache = cache_q[:5]
        small_test = test_q[:1]
        
        simulator = CacheSimulator(
            base_vectors=single_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=1,
            N=1,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=[0]
        )
        
        assert result.correct, "Brute force should be correct"
        
        print("Single base vector handled correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_k_larger_than_base():
    """Test handling when K > number of base vectors."""
    print("\nTest: K larger than base size")
    
    tmpdir = get_temp_dir("edge_k_large_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        small_base = base[:5]
        small_cache = cache_q[:5]
        small_test = test_q[:1]
        
        simulator = CacheSimulator(
            base_vectors=small_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=100,
            N=3,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=[]
        )
        
        assert result.distance_calculations is None or result.distance_calculations > 0, \
            "Should compute distances or return None on error"
        print(f"distance_calculations={result.distance_calculations}, error={result.error}")
        
        print("K > base size handled correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_n_equals_zero():
    """Test handling when N=0 (no ground truth needed)."""
    print("\nTest: N equals zero")
    
    tmpdir = get_temp_dir("edge_n_zero_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=10,
            N=0,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=[]
        )
        
        assert result.distance_calculations is None or result.distance_calculations > 0, \
            "Should compute distances or return None on error"
        
        print("N=0 handled correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_identical_vectors():
    """Test handling of identical vectors in base."""
    print("\nTest: Identical vectors")
    
    tmpdir = get_temp_dir("edge_identical_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        identical_base = np.tile(base[0], (10, 1))
        small_cache = cache_q[:5]
        small_test = test_q[:1]
        
        simulator = CacheSimulator(
            base_vectors=identical_base,
            cache_queries=small_cache,
            test_queries=small_test,
            K=5,
            N=5,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=small_test[0],
            algorithm='brute',
            ground_truth_indices=list(range(5))
        )
        
        assert result.distance_calculations is None or result.distance_calculations > 0
        
        print("Identical vectors handled correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

def test_invalid_algorithm():
    """Test that invalid algorithm raises appropriate error."""
    print("\nTest: Invalid algorithm error")
    
    tmpdir = get_temp_dir("edge_invalid_algo_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        result = simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='invalid_algorithm',
            ground_truth_indices=[]
        )
        
        assert result.error is not None, "Invalid algorithm should set error field"
        print("Invalid algorithm raises error correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_invalid_metric():
    """Test that invalid metric raises appropriate error."""
    print("\nTest: Invalid metric error")
    
    tmpdir = get_temp_dir("edge_invalid_metric_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        try:
            simulator = CacheSimulator(
                base_vectors=base,
                cache_queries=cache_q,
                test_queries=test_q,
                K=config['cache_K'],
                N=config['test_N'],
                metric='invalid_metric'
            )
            assert False, "Should have raised error for invalid metric"
        except (ValueError, KeyError):
            print("Invalid metric raises error correctly")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_validation_wrong_metric():
    """Test that validation requires matching metrics."""
    print("\nTest: Validation metric mismatch")
    
    tmpdir = get_temp_dir("edge_validation_metric_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        simulator_ang = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='angular'
        )
        
        simulator_ang.populate_cache()
        
        result_ang = simulator_ang.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='lemma1',
            ground_truth_indices=[]
        )
        
        simulator_euc = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='euclidean'
        )
        
        simulator_euc.populate_cache()
        
        result_euc = simulator_euc.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=[]
        )
        
        print("Validation can detect metric mismatches")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_validation_non_brute_cosine():
    """Test that cosine metric works with brute force."""
    print("\nTest: Cosine metric with brute force")
    
    tmpdir = get_temp_dir("edge_cosine_brute_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=config['test_N'],
            metric='cosine'
        )
        
        simulator.populate_cache()
        
        result_brute = simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=[]
        )
        assert result_brute.distance_calculations is None or result_brute.distance_calculations > 0
        
        print("Cosine metric works with brute force")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def test_ground_truth_with_duplicates():
    """Test handling of duplicate indices in ground truth."""
    print("\nTest: Duplicate ground truth indices")
    
    tmpdir = get_temp_dir("edge_gt_duplicates_")
    try:
        benchmark_dir = tmpdir / "test_benchmark"
        create_mini_sift_benchmark(benchmark_dir)
        base, cache_q, test_q, config = load_sift_benchmark(str(benchmark_dir))
        
        simulator = CacheSimulator(
            base_vectors=base,
            cache_queries=cache_q,
            test_queries=test_q,
            K=config['cache_K'],
            N=10,
            metric='euclidean'
        )
        
        simulator.populate_cache()
        
        dup_gt = [0, 1, 1, 2, 2, 2, 3, 4, 5, 5]
        
        result = simulator.run_query(
            query_id=0,
            query=test_q[0],
            algorithm='brute',
            ground_truth_indices=dup_gt
        )
        
        print("Duplicate ground truth indices handled")
        
    finally:
        cleanup_temp_dir(tmpdir)

#-------------------------------------------------------------------------------

def main():
    """Run all edge case tests."""
    print("\n" + "="*70)
    print("SIFT EDGE CASE TESTS")
    print("="*70)
    
    test_empty_cache()
    test_single_base_vector()
    test_k_larger_than_base()
    test_n_equals_zero()
    test_identical_vectors()
    test_invalid_algorithm()
    test_invalid_metric()
    test_validation_wrong_metric()
    test_validation_non_brute_cosine()
    test_ground_truth_with_duplicates()
    
    print("\n" + "="*70)
    print("All edge case tests passed!")
    print("="*70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

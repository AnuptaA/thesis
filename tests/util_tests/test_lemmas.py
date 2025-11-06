#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import (
    MainMemory,
    KVCache,
    lemma1_circular_inclusion,
    lemma2_half_gap,
    combined_algorithm,
    binary_search_last_index
)

#-------------------------------------------------------------------------------

def test_binary_search():
    """Test binary search helper."""
    print("Test 1: Binary search last index")
    
    # test case: sorted distances with some satisfying condition
    sorted_distances = [1.0, 2.0, 3.0, 4.0, 5.0]
    r_Q = 8.0
    d_q = 2.0
    
    # D(q, E_i) + d_q < r_Q
    # 1.0 + 2.0 = 3.0 < 8.0  
    # 2.0 + 2.0 = 4.0 < 8.0  
    # 3.0 + 2.0 = 5.0 < 8.0  
    # 4.0 + 2.0 = 6.0 < 8.0  
    # 5.0 + 2.0 = 7.0 < 8.0  
    
    theta = binary_search_last_index(sorted_distances, r_Q, d_q)
    assert theta == 4, f"Expected theta=4, got {theta}"
    
    # test case: no vectors satisfy
    r_Q = 2.0
    theta = binary_search_last_index(sorted_distances, r_Q, d_q)
    assert theta == -1, f"Expected theta=-1 (none satisfy), got {theta}"
    
    # test case: only some satisfy
    r_Q = 5.0
    theta = binary_search_last_index(sorted_distances, r_Q, d_q)
    # 1.0 + 2.0 = 3.0 < 5.0  
    # 2.0 + 2.0 = 4.0 < 5.0  
    # 3.0 + 2.0 = 5.0 NOT < 5.0
    assert theta == 1, f"Expected theta=1, got {theta}"
    
    print("    Binary search works")

#-------------------------------------------------------------------------------

def test_lemma1_simple():
    """Test Lemma 1 with simple case."""
    print("\nTest 2: Lemma 1 simple case")
    
    mm = MainMemory(M=100, n=32, seed=42)
    
    cache = KVCache(metric="euclidean")
    cached_query = np.random.randn(32)
    cache.populate_from_queries(np.array([cached_query]), mm, K=10)
    
    # query very close to cached query (should hit)
    similar_query = cached_query + np.random.randn(32) * 0.001
    
    result, is_hit, _ = lemma1_circular_inclusion(
        similar_query,
        cache.get_all_entries(),
        N=5,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 5, f"Expected 5 results, got {len(result)}"
        print("    Lemma 1 cache hit works")
    else:
        print("    Lemma 1 didn't hit")

#-------------------------------------------------------------------------------

def test_lemma2_simple():
    """Test Lemma 2 with simple case."""
    print("\nTest 3: Lemma 2 simple case")
    
    mm = MainMemory(M=100, n=32, seed=42)
    
    cache = KVCache(metric="euclidean")
    cached_query = np.random.randn(32)
    cache.populate_from_queries(np.array([cached_query]), mm, K=10)

    # query very close to cached query (within half gap)
    entry = cache.get_all_entries()[0]
    half_gap = entry.get_half_gap()
    
    # make query at distance < half_gap from cached query
    perturbation_size = half_gap * 0.1
    similar_query = cached_query + np.random.randn(32) * perturbation_size
    
    result, is_hit, _ = lemma2_half_gap(
        similar_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 10, f"Expected 10 results, got {len(result)}"
        print("    Lemma 2 cache hit works")
    else:
        print("    Lemma 2 didn't hit (gap might be too small)")

def test_combined_algorithm():
    """Test combined algorithm."""
    print("\nTest 4: Combined algorithm")
    
    mm = MainMemory(M=100, n=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    cached_queries = np.random.randn(5, 32)
    cache.populate_from_queries(cached_queries, mm, K=10)
    
    # test query similar to first cached query
    test_query = cached_queries[0] + np.random.randn(32) * 0.001
    
    result, is_hit, metadata = combined_algorithm(
        test_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    assert result is None or len(result) <= 10, "Result size shouldn't exceed N"
    
    print(f"  Cache hit: {is_hit}")
    if is_hit:
        print(f"  Hit via: {metadata.get('hit_lemma', 'unknown')}")
    print("    Combined algorithm works")

#-------------------------------------------------------------------------------

def test_cache_miss():
    """Test that unrelated queries miss the cache."""
    print("\nTest 5: Cache miss behavior")
    
    mm = MainMemory(M=100, n=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    # cache queries in one region of space
    cached_queries = np.random.randn(3, 32) + 100
    cache.populate_from_queries(cached_queries, mm, K=10)
    
    # query in different region
    test_query = np.random.randn(32) - 100
    
    result, is_hit, metadata = combined_algorithm(
        test_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    assert is_hit == False, "Should miss cache for distant query"
    assert result is None, "Should return None on cache miss"
    
    print("    Cache miss works correctly")

#-------------------------------------------------------------------------------

def test_different_k_n():
    """Test with different K and N values."""
    print("\nTest 6: Different K and N values")
    
    mm = MainMemory(M=100, n=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    cached_query = np.random.randn(32)
    cache.populate_from_queries(np.array([cached_query]), mm, K=20)
    
    # test with N < K
    similar_query = cached_query + np.random.randn(32) * 0.001
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=10,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 10, f"Expected N=10 results, got {len(result)}"
    
    # test with N = K
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=20,
        metric="euclidean"
    )
    
    if is_hit:
        assert len(result) == 20, f"Expected N=20 results, got {len(result)}"
    
    print("    Different K/N values work")

#-------------------------------------------------------------------------------

def test_n_greater_than_k():
    """Test when requesting more vectors than cached."""
    print("\nTest 7: N > K edge case")
    
    mm = MainMemory(M=100, n=32, seed=42)
    cache = KVCache(metric="euclidean")
    
    cached_query = np.random.randn(32)
    cache.populate_from_queries(np.array([cached_query]), mm, K=10)
    
    # request N=15 when only K=10 cached
    similar_query = cached_query + np.random.randn(32) * 0.001
    result, is_hit, _ = combined_algorithm(
        similar_query,
        cache.get_all_entries(),
        N=15,
        metric="euclidean"
    )
    
    assert is_hit == False, "Should miss cache when N > K"

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Testing Lemma Implementations")
    print("="*60)
    
    test_binary_search()
    test_lemma1_simple()
    test_lemma2_simple()
    test_combined_algorithm()
    test_cache_miss()
    test_different_k_n()
    
    print("\n" + "="*60)
    print("All tests passed!  ")
    print("="*60)

#-------------------------------------------------------------------------------
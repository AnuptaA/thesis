#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory, verify_cache_hit

#-------------------------------------------------------------------------------

def test_verify_correct_result():
    """Test verification of correct cache hit."""
    print("Test 1: Verify correct result")
    
    mm = MainMemory(M=100, n=32, seed=42)
    
    query = np.random.randn(32)
    true_top_k, _, _ = mm.top_k_search(query, k=10, metric="euclidean")
    
    is_correct, _ = verify_cache_hit(
        query,
        true_top_k,
        N=10,
        main_memory=mm,
        metric="euclidean"
    )
    
    assert is_correct, "Correct result should verify as correct"
    print("    Correct result verification works")

#-------------------------------------------------------------------------------

def test_verify_incorrect_result():
    """Test verification catches incorrect results."""
    print("\nTest 2: Verify incorrect result")
    
    mm = MainMemory(M=100, n=32, seed=42)
    
    query = np.random.randn(32)
    true_top_k, _, _ = mm.top_k_search(query, k=10, metric="euclidean")
    
    # corrupt one result (replace with random vector)
    corrupted_result = true_top_k.copy()
    corrupted_result[5] = np.random.randn(32)
    
    is_correct, _ = verify_cache_hit(
        query,
        corrupted_result,
        N=10,
        main_memory=mm,
        metric="euclidean"
    )
    
    assert not is_correct, "Corrupted result should verify as incorrect"
    print("    Incorrect result detection works")

#-------------------------------------------------------------------------------

def test_verify_partial_result():
    """Test verification with partial overlap."""
    print("\nTest 3: Verify partial result")
    
    mm = MainMemory(M=100, n=32, seed=42)
    
    query = np.random.randn(32)
    true_top_10, _, _ = mm.top_k_search(query, k=10, metric="euclidean")
    
    # take only first 5 (which are correct)
    partial_result = true_top_10[:5]
    
    is_correct, _ = verify_cache_hit(
        query,
        partial_result,
        N=5,
        main_memory=mm,
        metric="euclidean"
    )
    
    assert is_correct, "Partial correct result should verify"
    print("    Partial result verification works")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Testing Verification Functions")
    print("="*60)
    
    test_verify_correct_result()
    test_verify_incorrect_result()
    test_verify_partial_result()
    
    print("\n" + "="*60)
    print("All tests passed!  ")
    print("="*60)

#-------------------------------------------------------------------------------
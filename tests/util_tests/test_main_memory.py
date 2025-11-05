#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory

#-------------------------------------------------------------------------------

def test_random_initialization():
    """Test generating random vectors."""
    print("Test 1: Random initialization")
    mm = MainMemory(M=100, n=64, seed=42)
    
    assert mm.M == 100, "Wrong M"
    assert mm.n == 64, "Wrong n"
    assert len(mm.vectors) == 100, "Wrong number of vectors"
    assert mm.vectors[0].shape[0] == 64, "Wrong dimension"
    print("  > Random initialization works")

#-------------------------------------------------------------------------------

def test_from_vectors():
    """Test initializing from existing vectors."""
    print("\nTest 2: Initialize from existing vectors")
    
    # create some vectors
    existing_vecs = [np.random.randn(32) for _ in range(50)]
    
    mm = MainMemory(vectors=existing_vecs)
    
    assert mm.M == 50, "Wrong M"
    assert mm.n == 32, "Wrong n"
    assert len(mm.vectors) == 50, "Wrong number of vectors"
    assert np.array_equal(mm.vectors[0], existing_vecs[0]), "Vectors don't match"
    print("  > Initialization from vectors works")

#-------------------------------------------------------------------------------

def test_top_k_search():
    """Test top-k search."""
    print("\nTest 3: Top-k search")
    
    mm = MainMemory(M=1000, n=128, seed=42)
    query = np.random.randn(128)
    
    top_k_vecs, top_k_dists, gap = mm.top_k_search(query, k=10, metric="euclidean")
    
    assert len(top_k_vecs) == 10, "Wrong number of results"
    assert len(top_k_dists) == 10, "Wrong number of distances"
    assert gap >= 0, "Gap should be non-negative"
    
    # check distances are sorted
    for i in range(len(top_k_dists) - 1):
        assert top_k_dists[i] <= top_k_dists[i+1], "Distances not sorted"
    
    print("  > Top-k search works")

#-------------------------------------------------------------------------------

def test_missing_args():
    """Test that missing arguments raise error."""
    print("\nTest 4: Missing arguments")
    
    try:
        mm = MainMemory()  # should fail
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  > Correctly raised error: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Testing MainMemory")
    print("="*60)
    
    test_random_initialization()
    test_from_vectors()
    test_top_k_search()
    test_missing_args()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

#-------------------------------------------------------------------------------
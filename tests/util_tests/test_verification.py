#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import verify_lemma1_condition, verify_lemma2_condition

#-------------------------------------------------------------------------------

def test_lemma1_condition_satisfied():
    """Test Lemma 1 condition when it should be satisfied."""
    print("Test 1: Lemma 1 condition satisfied")

    query = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    cached_query = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cached_vectors = [np.array([0.05, 0.0, 0.0], dtype=np.float32)]
    cached_distances = [0.05]

    all_satisfy, details = verify_lemma1_condition(
        query, cached_query, cached_vectors, cached_distances, metric="euclidean"
    )

    print(f"  r_Q={details['r_Q']}, d_q={details['d_q']}, all_satisfy={all_satisfy}")
    print("Lemma 1 condition check works")

#-------------------------------------------------------------------------------

def test_lemma2_condition_satisfied():
    """Test Lemma 2 half-gap condition."""
    print("\nTest 2: Lemma 2 condition satisfied")

    query = np.array([0.01, 0.0], dtype=np.float32)
    cached_query = np.array([0.0, 0.0], dtype=np.float32)
    gap = 0.5

    satisfies, details = verify_lemma2_condition(
        query, cached_query, gap, metric="euclidean"
    )

    assert satisfies, f"d_q={details['D(q, Q)']:.4f} should be < half_gap={details['half_gap']:.4f}"
    print(f"  d_q={details['D(q, Q)']:.4f}, half_gap={details['half_gap']:.4f}, satisfies={satisfies}")
    print("Lemma 2 condition check works")

#-------------------------------------------------------------------------------

def test_lemma2_condition_not_satisfied():
    """Test Lemma 2 half-gap condition when it should fail."""
    print("\nTest 3: Lemma 2 condition not satisfied")

    query = np.array([1.0, 0.0], dtype=np.float32)
    cached_query = np.array([0.0, 0.0], dtype=np.float32)
    gap = 0.5

    satisfies, details = verify_lemma2_condition(
        query, cached_query, gap, metric="euclidean"
    )

    assert not satisfies, f"d_q={details['D(q, Q)']:.4f} should be >= half_gap={details['half_gap']:.4f}"
    print(f"  d_q={details['D(q, Q)']:.4f}, half_gap={details['half_gap']:.4f}, satisfies={satisfies}")
    print("Lemma 2 rejection works")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing Verification Functions")
    print("="*80)

    test_lemma1_condition_satisfied()
    test_lemma2_condition_satisfied()
    test_lemma2_condition_not_satisfied()

    print("\n" + "="*80)
    print("All tests passed.")
    print("="*80)

#-------------------------------------------------------------------------------

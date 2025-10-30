#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple
from .distance_metrics import get_distance_function, DistanceMetric
from .main_memory import MainMemory

#-------------------------------------------------------------------------------

def verify_cache_hit(
    query: np.ndarray,
    cached_result: List[np.ndarray],
    N: int,
    main_memory: MainMemory,
    metric: DistanceMetric = "euclidean",
    tolerance: float = 1e-6
) -> Tuple[bool, dict]:
    """
    Verify that a cache hit is truly correct by computing the ground truth.
    
    Checks that the cached result contains exactly the same vectors as the
    ground truth top-N search (allowing for floating point tolerance).
    
    Args:
        query: Query vector
        cached_result: Result returned by the cache algorithm
        N: Number of top results requested
        main_memory: MainMemory object to compute ground truth
        metric: Distance metric used
        tolerance: Numerical tolerance for distance comparisons
        
    Returns:
        Tuple of (is_correct, details)
        - is_correct: Whether the cached result matches ground truth
        - details: Dictionary with verification information
    """
    if cached_result is None or len(cached_result) == 0:
        return False, {"error": "No cached result provided"}
    
    distance_func = get_distance_function(metric)
    
    # compute ground truth
    ground_truth, ground_truth_distances, _ = main_memory.top_k_search(query, N, metric)
    
    # compute distances for cached result
    cached_distances = [distance_func(query, vec) for vec in cached_result]
    
    # sort both for comparison
    sorted_ground_truth_distances = sorted(ground_truth_distances)
    sorted_cached_distances = sorted(cached_distances)
    
    # check lengths match
    is_correct = len(cached_result) == len(ground_truth) == N
    
    if not is_correct:
        details = {
            "error": "Length mismatch",
            "expected_length": N,
            "cached_length": len(cached_result),
            "ground_truth_length": len(ground_truth),
            "correct": False
        }
        return False, details
    
    # check distances match within tolerance
    max_diff = 0.0
    for gt_dist, cached_dist in zip(sorted_ground_truth_distances, sorted_cached_distances):
        diff = abs(gt_dist - cached_dist)
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            is_correct = False
            break
    
    details = {
        "ground_truth_distances": sorted_ground_truth_distances,
        "cached_distances": sorted_cached_distances,
        "max_distance_diff": max_diff,
        "tolerance": tolerance,
        "correct": is_correct
    }
    
    return is_correct, details

#-------------------------------------------------------------------------------

def verify_lemma1_condition(
    query: np.ndarray,
    cached_query: np.ndarray,
    cached_vectors: List[np.ndarray],
    cached_distances: List[float],
    metric: DistanceMetric = "euclidean"
) -> Tuple[bool, dict]:
    """
    Verify that Lemma 1 circular inclusion condition holds.
    
    Args:
        query: New query vector q
        cached_query: Cached query vector Q
        cached_vectors: Top-K vectors from cache
        cached_distances: Distances from Q to cached vectors
        metric: Distance metric used
        
    Returns:
        Tuple of (all_satisfy, details)
        - all_satisfy: Whether all vectors satisfy the condition
        - details: Dictionary with per-vector verification info
    """
    distance_func = get_distance_function(metric)
    
    K = len(cached_vectors)
    if K == 0:
        return True, {"error": "No cached vectors"}
    
    r_Q = cached_distances[-1]  # distance to K-th vector
    d_q = distance_func(query, cached_query)
    
    results = []
    all_satisfy = True
    
    for i, (vec, dist_Q_to_E) in enumerate(zip(cached_vectors, cached_distances)):
        dist_q_to_E = distance_func(query, vec)
        condition_value = dist_q_to_E + d_q
        satisfies = condition_value < r_Q
        
        results.append({
            "index": i,
            "D(q, E_i)": dist_q_to_E,
            "D(q, Q)": d_q,
            "D(Q, E_i)": dist_Q_to_E,
            "D(q, E_i) + D(q, Q)": condition_value,
            "r_Q": r_Q,
            "satisfies": satisfies
        })
        
        if not satisfies:
            all_satisfy = False
    
    details = {
        "r_Q": r_Q,
        "d_q": d_q,
        "K": K,
        "all_satisfy": all_satisfy,
        "per_vector_results": results
    }
    
    return all_satisfy, details

#-------------------------------------------------------------------------------

def verify_lemma2_condition(
    query: np.ndarray,
    cached_query: np.ndarray,
    gap: float,
    metric: DistanceMetric = "euclidean"
) -> Tuple[bool, dict]:
    """
    Verify that Lemma 2 half-gap condition holds.
    
    Args:
        query: New query vector q
        cached_query: Cached query vector Q
        gap: Gap value between K-th and (K+1)-th vectors
        metric: Distance metric used
        
    Returns:
        Tuple of (satisfies, details)
        - satisfies: Whether the condition is satisfied
        - details: Dictionary with verification info
    """
    distance_func = get_distance_function(metric)
    
    d_q = distance_func(query, cached_query)
    half_gap = gap / 2.0
    satisfies = d_q < half_gap
    
    details = {
        "D(q, Q)": d_q,
        "gap": gap,
        "half_gap": half_gap,
        "satisfies": satisfies,
        "margin": half_gap - d_q if satisfies else d_q - half_gap
    }
    
    return satisfies, details
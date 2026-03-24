#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple
from .distance_metrics import get_distance_function, DistanceMetric

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
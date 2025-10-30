#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple, Optional
from .distance_metrics import get_distance_function, DistanceMetric
from .kv_cache import CacheEntry

#-------------------------------------------------------------------------------

def binary_search_last_index(
    sorted_distances: List[float],
    r_Q: float,
    d_q: float
) -> int:
    """
    Binary search to find the last index where D(q, E_i) + d_q < r_Q.
    
    Since sorted_distances is sorted in ascending order by D(q, E_i),
    this finds the rightmost index satisfying the circular inclusion condition.
    
    Args:
        sorted_distances: Distances from q to each vector, sorted ascending
        r_Q: Cache radius (distance from Q to its K-th result)
        d_q: Distance from q to Q
        
    Returns:
        Last index satisfying the condition, or -1 if none satisfy
    """
    left, right = 0, len(sorted_distances) - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if sorted_distances[mid] + d_q < r_Q:
            # found vector satisfying lemma, try to find a later index
            result = mid
            left = mid + 1
        else:
            # not satisfied, search earlier
            right = mid - 1
    
    return result

#-------------------------------------------------------------------------------

def lemma1_circular_inclusion(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean"
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Lemma 1 - Circular Inclusion Guarantee algorithm.
    
    Args:
        query: New query vector q
        cache_entries: List of cache entries
        N: Desired number of top-N results
        metric: Distance metric to use
        
    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata)
    """
    distance_func = get_distance_function(metric)
    R_q = []
    R_q_set = set()
    
    metadata = {
        "checked_entries": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "theta_values": []
    }
    
    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        
        Q = entry.query
        R_Q = entry.top_k_vectors

        r_Q = entry.get_radius()        # D(Q, E_K)
        d_q = distance_func(query, Q)   # D(q, Q)
        
        # sort R_Q by distance to query q
        distances_to_q = [distance_func(query, E_i) for E_i in R_Q]
        sorted_indices = np.argsort(distances_to_q)
        sorted_R_Q = [R_Q[i] for i in sorted_indices]
        sorted_distances = [distances_to_q[i] for i in sorted_indices]
        
        # find last index using binary search
        theta = binary_search_last_index(sorted_distances, r_Q, d_q)
        metadata["theta_values"].append(theta)
        
        # unite sets and handle duplicates
        for i in range(theta + 1):
            vec_bytes = sorted_R_Q[i].tobytes()
            if vec_bytes not in R_q_set:
                R_q.append(sorted_R_Q[i])
                R_q_set.add(vec_bytes)
                metadata["vectors_added"] += 1
        
        # check set size condition
        if len(R_q) >= N:
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            
            # re-sort R_q by distance to query and return exactly N
            distances = [distance_func(query, vec) for vec in R_q]
            sorted_indices = np.argsort(distances)
            result = [R_q[i] for i in sorted_indices[:N]]
            
            return result, True, metadata
    
    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

def lemma2_half_gap(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean"
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Implement Lemma 2 - Half-Gap Guarantee algorithm.
    
    Args:
        query: New query vector q
        cache_entries: List of cache entries
        N: Desired number of top-N results
        metric: Distance metric to use
        
    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata)
    """
    distance_func = get_distance_function(metric)
    
    metadata = {
        "checked_entries": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "d_q": None,
        "half_gap": None,
        "condition_satisfied": False
    }
    
    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        
        Q = entry.query
        R_Q = entry.top_k_vectors
        K = len(R_Q)
        
        d_q = distance_func(query, Q)
        half_gap = entry.get_half_gap()
        
        metadata["d_q"] = d_q
        metadata["half_gap"] = half_gap
        
        # check half-gap condition
        if d_q < half_gap:
            metadata["condition_satisfied"] = True
            
            # sort R_Q by distance to query
            distances_to_q = [distance_func(query, E_i) for E_i in R_Q]
            sorted_indices = np.argsort(distances_to_q)
            sorted_R_Q = [R_Q[i] for i in sorted_indices]
            
            if K == N:
                # exact match
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                return sorted_R_Q, True, metadata
            
            elif K > N:
                # return top-N
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                return sorted_R_Q[:N], True, metadata

    # cache miss
    return None, False, metadata

#-------------------------------------------------------------------------------

def combined_algorithm(
    query: np.ndarray,
    cache_entries: List[CacheEntry],
    N: int,
    metric: DistanceMetric = "euclidean"
) -> Tuple[Optional[List[np.ndarray]], bool, dict]:
    """
    Combined algorithm: Try Lemma 2 first, then fallback to Lemma 1.

    Args:
        query: New query vector q
        cache_entries: List of cache entries
        N: Desired number of top-N results
        metric: Distance metric to use
        
    Returns:
        Tuple of (result_vectors, is_cache_hit, metadata)
    """
    distance_func = get_distance_function(metric)
    R_q = []
    R_q_set = set()
    
    metadata = {
        "checked_entries": 0,
        "lemma2_attempts": 0,
        "lemma2_successes": 0,
        "lemma1_attempts": 0,
        "vectors_added": 0,
        "cache_hit": False,
        "hit_entry_index": None,
        "hit_lemma": None
    }
    
    for entry_idx, entry in enumerate(cache_entries):
        metadata["checked_entries"] += 1
        
        Q = entry.query
        R_Q = entry.top_k_vectors
        K = len(R_Q)
        
        d_q = distance_func(query, Q)   # D(q, Q)
        
        # sort R_Q by distance to query
        distances_to_q = [distance_func(query, E_i) for E_i in R_Q]
        sorted_indices = np.argsort(distances_to_q)
        sorted_R_Q = [R_Q[i] for i in sorted_indices]
        sorted_distances = [distances_to_q[i] for i in sorted_indices]
        
        # try Lemma 2 first
        half_gap = entry.get_half_gap()
        if d_q < half_gap:
            metadata["lemma2_attempts"] += 1
            
            if K == N:
                # exact match - cache hit
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                metadata["hit_lemma"] = "lemma2"
                metadata["lemma2_successes"] += 1
                return sorted_R_Q, True, metadata
            
            elif K > N:
                # return top-N - cache hit
                metadata["cache_hit"] = True
                metadata["hit_entry_index"] = entry_idx
                metadata["hit_lemma"] = "lemma2"
                metadata["lemma2_successes"] += 1
                return sorted_R_Q[:N], True, metadata
            
            else:  # K < N
                # add all of R_Q to R_q and continue
                for i in range(K):
                    vec_bytes = sorted_R_Q[i].tobytes()
                    if vec_bytes not in R_q_set:
                        R_q.append(sorted_R_Q[i])
                        R_q_set.add(vec_bytes)
                        metadata["vectors_added"] += 1
                continue
        
        # fallback to Lemma 1
        metadata["lemma1_attempts"] += 1
        r_Q = entry.get_radius()  # D(Q, E_K)
        
        # find last index using binary search
        theta = binary_search_last_index(sorted_distances, r_Q, d_q)
        
        # unite sets and handle duplicates
        for i in range(theta + 1):
            vec_bytes = sorted_R_Q[i].tobytes()
            if vec_bytes not in R_q_set:
                R_q.append(sorted_R_Q[i])
                R_q_set.add(vec_bytes)
                metadata["vectors_added"] += 1

        # check set size condition
        if len(R_q) >= N:
            metadata["cache_hit"] = True
            metadata["hit_entry_index"] = entry_idx
            metadata["hit_lemma"] = "lemma1"

            # sort R_q by distance to query
            distances = [distance_func(query, vec) for vec in R_q]
            sorted_indices = np.argsort(distances)
            result = [R_q[i] for i in sorted_indices[:N]]
            
            return result, True, metadata
    
    # cache miss
    return None, False, metadata
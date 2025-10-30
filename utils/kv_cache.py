#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List
from .distance_metrics import DistanceMetric, get_distance_function

#-------------------------------------------------------------------------------

class CacheEntry:
    """
    Represents a single cache entry.
    """
    
    def __init__(
        self, 
        query: np.ndarray, 
        top_k_vectors: List[np.ndarray],
        top_k_distances: List[float],
        gap: float
    ):
        """
        Initialize a cache entry.
        
        Args:
            query: Query vector (key)
            top_k_vectors: Top-K closest vectors
            top_k_distances: Distances to top-K vectors
            gap: Gap between K-th and (K+1)-th vector
        """
        self.query = query
        self.top_k_vectors = top_k_vectors
        self.top_k_distances = top_k_distances
        self.gap = gap
        self.k = len(top_k_vectors)
    
    def get_radius(self) -> float:
        """
        Get the cache radius (distance to the K-th vector).
        
        Returns:
            Cache radius r_Q
        """
        return self.top_k_distances[-1] if self.top_k_distances else 0.0
    
    def get_half_gap(self) -> float:
        """
        Get half of the gap.
        
        Returns:
            Half gap value
        """
        return self.gap / 2.0

#-------------------------------------------------------------------------------

class KVCache:
    """
    Key-Value cache for storing query results.
    Keys are query embeddings, values are (top_k_vectors, gap/2).
    """
    
    def __init__(self, metric: DistanceMetric = "euclidean"):
        """
        Initialize the KV cache.
        
        Args:
            metric: Distance metric to use
        """
        self.cache: List[CacheEntry] = []
        self.metric = metric
        self.distance_func = get_distance_function(metric)
    
    def add_entry(
        self,
        query: np.ndarray,
        top_k_vectors: List[np.ndarray],
        top_k_distances: List[float],
        gap: float
    ):
        """
        Add a new entry to the cache.
        
        Args:
            query: Query vector
            top_k_vectors: Top-K closest vectors
            top_k_distances: Distances to top-K vectors
            gap: Gap between K-th and (K+1)-th vector
        """
        entry = CacheEntry(query, top_k_vectors, top_k_distances, gap)
        self.cache.append(entry)
    
    def get_all_entries(self) -> List[CacheEntry]:
        """
        Get all cache entries.
        
        Returns:
            List of cache entries
        """
        return self.cache
    
    def clear(self):
        """
        Clear all cache entries.
        """
        self.cache.clear()
    
    def size(self) -> int:
        """
        Get the number of entries in the cache.
        
        Returns:
            Number of cache entries
        """
        return len(self.cache)
    
    def __repr__(self) -> str:
        return f"KVCache(metric={self.metric}, size={self.size()})"

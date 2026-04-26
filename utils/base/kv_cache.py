#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Optional
from .distance_metrics import DistanceMetric
from .main_memory import MainMemory

#-------------------------------------------------------------------------------

class CacheEntry:
    """
    Represents a single cache entry.
    """

    def __init__(
        self,
        query: np.ndarray,
        top_k_vectors: List[np.ndarray],
        r_Q: float,
        half_gap: float,
        top_k_indices: Optional[List[int]] = None
    ):
        """
        Initialize a cache entry.

        Args:
            query: Query vector (key)
            top_k_vectors: Top-K closest vectors
            r_Q: Distance to the K-th (farthest cached) vector
            half_gap: Half of the difference between K-th and (K+1)-th vector
            top_k_indices: Stable vector indices corresponding to top_k_vectors
        """
        self.query = query
        self.top_k_vectors = top_k_vectors
        self.r_Q = r_Q
        self.half_gap = half_gap
        self.top_k_indices = top_k_indices

    def get_radius(self) -> float:
        """
        Get the distance to the K-th (farthest cached) vector.

        Returns:
            Caching radius (r_Q)
        """
        return self.r_Q
    
    def get_half_gap(self) -> float:
        """
        Get half of the difference between K-th and (K+1)-th vector.
        
        Returns:
            Half gap value (half_gap)
        """
        return self.half_gap

#-------------------------------------------------------------------------------

class KVCache:
    """
    Key-Value cache for storing query results.
    Keys are query embeddings, values are (top_k_vectors, radius, half_gap).
    """
    
    def __init__(self, metric: DistanceMetric = "euclidean"):
        """
        Initialize the KV cache.
        
        Args:
            metric: Distance metric to use
        """
        self.cache: List[CacheEntry] = []
        self.metric = metric
    
    def add_entry(
        self,
        query: np.ndarray,
        top_k_vectors: List[np.ndarray],
        r_Q: float,
        half_gap: float,
        top_k_indices: Optional[List[int]] = None
    ):
        """
        Add a new entry to the cache.

        Args:
            query: Query vector
            top_k_vectors: Top-K closest vectors
            r_Q: Distance to the K-th (farthest cached) vector
            half_gap: Half of the difference between K-th and (K+1)-th vector
            top_k_indices: Stable vector indices corresponding to top_k_vectors
        """
        entry = CacheEntry(query, top_k_vectors, r_Q, half_gap, top_k_indices)
        self.cache.append(entry)

    def populate_from_queries(
        self,
        queries: np.ndarray,
        main_memory: MainMemory,
        K: int
    ):
        """
        Populate cache by running brute-force top-K search for each query.
        
        Args:
            queries: Array of query vectors to cache
            main_memory: MainMemory instance to search
            K: Number of vectors to cache per query
        """
        for query in queries:
            top_k_vecs, top_k_dists, gap, top_k_indices = main_memory.top_k_search(
                query,
                K,
                self.metric,
                return_indices=True
            )
            self.add_entry(query, top_k_vecs, float(top_k_dists[-1]), gap / 2.0, top_k_indices)
    
    def get_all_entries(self) -> List[CacheEntry]:
        """
        Get all cache entries.
        
        Returns:
            List of cache entries
        """
        return self.cache
    
    def clear(self) -> None:
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

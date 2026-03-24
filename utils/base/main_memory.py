#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple, Optional
from .distance_metrics import DistanceMetric, get_distance_function

#-------------------------------------------------------------------------------

class MainMemory:
    """
    Represents main memory containing M vectors of dimension N.
    """

    def __init__(
        self, 
        M: int = None, 
        N: int = None, 
        seed: int = None,
        vectors: List[np.ndarray] = None
    ):
        """
        Initialize main memory.
        
        Args:
            M: Number of vectors, required if vectors not provided
            N: Dimension of each vector, required if vectors not provided
            seed: Random seed for reproducibility
            vectors: Pre-existing vectors to load (optional)
        """
        if vectors is not None:
            self.vectors = vectors
            self.M = len(vectors)
            self.N = vectors[0].shape[0]
            self.seed = None
        else:
            # dummy testing, generate random
            if M is None or N is None:
                raise ValueError("M and N must be provided if vectors is None")
            
            self.M = M
            self.N = N
            self.seed = seed
            
            if seed is not None:
                np.random.seed(seed)
                print("Seed used for main memory:", seed)
            
            self.vectors = self._generate_random_vectors()

    def _generate_random_vectors(self) -> List[np.ndarray]:
        """Generate M random vectors of dimension N."""
        vectors = []
        for _ in range(self.M):
            vec = np.random.randn(self.N)
            vectors.append(vec)
        return vectors
        
    def top_k_search(
        self, 
        query: np.ndarray, 
        k: int, 
        metric: DistanceMetric = "euclidean",
        distance_tracker: Optional[callable] = None
    ) -> Tuple[List[np.ndarray], List[float], float]:
        """
        Find the top-k closest vectors to the query.
        
        Args:
            query: Query vector
            k: Number of closest vectors to return
            metric: Distance metric
            distance_tracker: Optional function to track distance calculations
            
        Returns:
            Tuple of (top_k_vectors, top_k_distances, gap)
        """
        distance_func = get_distance_function(metric)
        if distance_tracker is not None:
            distance_func = distance_tracker
        
        distances = []
        for i, vec in enumerate(self.vectors):
            dist = distance_func(query, vec)
            distances.append((dist, i))
        
        # sort by distance and get top k
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        
        top_k_vectors = [self.vectors[i] for _, i in top_k]
        top_k_distances = [d for d, _ in top_k]
        
        # compute gap
        if len(distances) > k:
            gap = distances[k][0] - distances[k-1][0]
        else:
            gap = 0.0
        
        return top_k_vectors, top_k_distances, gap
    
    def get_all_vectors(self) -> List[np.ndarray]:
        """
        Get all vectors in main memory.
        
        Returns:
            List of all vectors
        """
        return self.vectors
    
    def __repr__(self) -> str:
        return f"MainMemory(M={self.M}, N={self.N})"

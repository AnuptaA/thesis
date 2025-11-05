#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import List, Tuple
from .distance_metrics import DistanceMetric, get_distance_function

#-------------------------------------------------------------------------------

class MainMemory:
    """
    Represents main memory containing M vectors of dimension n.
    """

    def __init__(
        self, 
        M: int = None, 
        n: int = None, 
        seed: int = None,
        vectors: List[np.ndarray] = None
    ):
        """
        Initialize main memory.
        
        Args:
            M: Number of vectors, required if vectors not provided
            n: Dimension of each vector (qrequired if vectors not provided)
            seed: Random seed for reproducibility
            vectors: Pre-existing vectors to load (optional)
        """
        if vectors is not None:
            self.vectors = vectors
            self.M = len(vectors)
            self.n = vectors[0].shape[0]
            self.seed = None
        else:
            # dummy testing, generate random
            if M is None or n is None:
                raise ValueError("M and n must be provided if vectors is None")
            
            self.M = M
            self.n = n
            self.seed = seed
            
            if seed is not None:
                np.random.seed(seed)
                print("Seed used for main memory:", seed)
            
            self.vectors = self._generate_random_vectors()

    def _generate_random_vectors(self) -> List[np.ndarray]:
        """Generate M random vectors of dimension n."""
        vectors = []
        for _ in range(self.M):
            vec = np.random.randn(self.n)
            vectors.append(vec)
        return vectors
        
    def top_k_search(
        self,
        query: np.ndarray,
        k: int,
        metric: DistanceMetric = "euclidean"
    ) -> Tuple[List[np.ndarray], List[float], float]:
        """
        Perform top-k search to find k closest vectors to query.
        
        Args:
            query: Query vector
            k: Number of top results to return
            metric: Distance metric to use
            
        Returns:
            Tuple of (top_k_vectors, top_k_distances, gap)
            - top_k_vectors: List of k closest vectors
            - top_k_distances: Distances to these vectors
            - gap: Distance difference between k-th and (k+1)-th vectors
        """
        if k > self.M:
            raise ValueError(f"k ({k}) cannot be larger than M ({self.M})")
        
        distance_func = get_distance_function(metric)
        
        # compute distances to all vectors
        distances = [(distance_func(query, vec), vec) for vec in self.vectors]
        
        # sort by distance
        distances.sort(key=lambda x: x[0])
        
        # extract top-k
        top_k_vectors = [vec for _, vec in distances[:k]]
        top_k_distances = [dist for dist, _ in distances[:k]]
        
        # compute gap (if k+1 exists)
        if k < self.M:
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
        return f"MainMemory(M={self.M}, n={self.n})"

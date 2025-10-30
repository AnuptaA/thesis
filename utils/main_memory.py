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
    
    def __init__(self, M: int, n: int, seed: int = None):
        """
        Initialize main memory with M random vectors of dimension n.
        
        Args:
            M: Number of vectors in main memory
            n: Dimension of each vector
            seed: Random seed for reproducibility
        """
        self.M = M
        self.n = n
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        self.vectors = self._generate_random_vectors()
    
    def _generate_random_vectors(self) -> List[np.ndarray]:
        """
        Generate M random vectors of dimension n.
        
        Returns:
            List of random vectors
        """
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
    
    def generate_similar_query(
        self, 
        base_vector: np.ndarray, 
        perturbation: float = 0.1
    ) -> np.ndarray:
        """
        Generate a query similar to a base vector by adding small noise.
        
        Args:
            base_vector: Base vector to perturb
            perturbation: Standard deviation of noise (relative to vector norm)
            
        Returns:
            Perturbed query vector (normalized)
        """
        noise = np.random.randn(self.n) * perturbation
        query = base_vector + noise

        # normalize
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        return query
    
    def generate_random_query(self) -> np.ndarray:
        """
        Generate a completely random query vector.
        
        Returns:
            Random normalized query vector
        """
        return self._generate_random_vectors(1, self.n)[0]
    
    def __repr__(self) -> str:
        return f"MainMemory(M={self.M}, n={self.n})"

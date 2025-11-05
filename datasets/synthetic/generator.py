#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict

#-------------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    name: str
    num_base_vectors: int
    num_cache_queries: int
    num_test_queries: int
    dimension: int
    K: int  # vectors cached per query
    N: int  # vectors requested per test query
    seed: Optional[int] = None
    
    # generate queries in different ways
    test_query_strategy: str = "mixed"  # "random", "similar", "clustered", "mixed"
    similarity_perturbation: float = 0.01  # for "similar" queries
    num_clusters: int = 5  # for "clustered" queries
    
    def to_dict(self) -> Dict:
        return asdict(self)

#-------------------------------------------------------------------------------

class SyntheticDatasetGenerator:
    """Generate synthetic datasets for lemma verification."""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize generator with configuration.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        self.base_vectors: Optional[np.ndarray] = None
        self.cache_queries: Optional[np.ndarray] = None
        self.test_queries: Optional[np.ndarray] = None
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete dataset.
        
        Returns:
            Tuple of (base_vectors, cache_queries, test_queries)
        """
        print(f"Generating dataset: {self.config.name}")
        print(f"  Base vectors: {self.config.num_base_vectors}")
        print(f"  Cache queries: {self.config.num_cache_queries}")
        print(f"  Test queries: {self.config.num_test_queries}")
        print(f"  Dimension: {self.config.dimension}")
        print(f"  K={self.config.K}, N={self.config.N}")
        
        # generate database vectors
        self.base_vectors = self._generate_base_vectors()
        
        # generate pre-populated cache queries
        self.cache_queries = self._generate_cache_queries()
        
        # generate test queries based on strategy
        self.test_queries = self._generate_test_queries()
        
        return self.base_vectors, self.cache_queries, self.test_queries
    
    def _generate_base_vectors(self) -> np.ndarray:
        """Generate base database vectors."""
        print("  Generating base vectors...")
        vectors = np.random.randn(
            self.config.num_base_vectors, 
            self.config.dimension
        ).astype(np.float32)
        return vectors
    
    def _generate_cache_queries(self) -> np.ndarray:
        """Generate queries for cache population."""
        print("  Generating cache queries...")
        queries = np.random.randn(
            self.config.num_cache_queries,
            self.config.dimension
        ).astype(np.float32)
        
        return queries
    
    def _generate_test_queries(self) -> np.ndarray:
        """Generate test queries based on strategy."""
        print(f"  Generating test queries (strategy: {self.config.test_query_strategy})...")
        
        strategy = self.config.test_query_strategy
        
        if strategy == "random":
            return self._generate_random_test_queries()
        elif strategy == "similar":
            return self._generate_similar_test_queries()
        elif strategy == "clustered":
            return self._generate_clustered_test_queries()
        elif strategy == "mixed":
            return self._generate_mixed_test_queries()
        else:
            raise ValueError(f"Unknown test query strategy: {strategy}")
    
    def _generate_random_test_queries(self) -> np.ndarray:
        """Generate completely random test queries."""
        return np.random.randn(
            self.config.num_test_queries,
            self.config.dimension
        ).astype(np.float32)
    
    def _generate_similar_test_queries(self) -> np.ndarray:
        """
        Generate test queries similar to cached queries. Perturb cached
        queries slightly to create "new" queries.
        """
        queries = []
        pert = self.config.similarity_perturbation
        
        for _ in range(self.config.num_test_queries):
            cache_idx = np.random.randint(0, self.config.num_cache_queries)
            base_query = self.cache_queries[cache_idx]
            
            # add small perturbation
            noise = np.random.randn(self.config.dimension) * pert
            queries.append(base_query + noise)
        
        return np.array(queries, dtype=np.float32)
    
    def _generate_clustered_test_queries(self) -> np.ndarray:
        """
        Generate test queries clustered around cache queries.
        """
        queries = []
        pert = self.config.similarity_perturbation
        queries_per_cluster = self.config.num_test_queries // self.config.num_clusters

        # collect all unique cluster centers
        cluster_indices = set()
        num_centers = min(self.config.num_clusters, self.config.num_cache_queries)
        while len(cluster_indices) < num_centers:
            center_idx = np.random.randint(0, self.config.num_cache_queries)
            cluster_indices.add(center_idx)
        
        # generate queries around each cluster center
        for center_idx in cluster_indices:
            center = self.cache_queries[center_idx]
            for _ in range(queries_per_cluster):
                noise = np.random.randn(self.config.dimension) * pert
                queries.append(center + noise)
        
        # fill remaining queries if any
        remaining = self.config.num_test_queries - len(queries)
        if remaining > 0:
            for _ in range(remaining):
                query = np.random.randn(self.config.dimension)
                queries.append(query)
        
        return np.array(queries, dtype=np.float32)
    
    def _generate_mixed_test_queries(self) -> np.ndarray:
        """
        Generate mix of random, similar, and clustered queries.
        
        Distribution:
        - 40% similar to cached queries
        - 30% clustered around cached queries  
        - 30% completely random
        """
        num_similar = int(self.config.num_test_queries * 0.4)
        num_clustered = int(self.config.num_test_queries * 0.3)
        num_random = self.config.num_test_queries - num_similar - num_clustered
        
        queries = []
        
        # similar queries
        for _ in range(num_similar):
            cache_idx = np.random.randint(0, self.config.num_cache_queries)
            base_query = self.cache_queries[cache_idx]
            noise = np.random.randn(self.config.dimension) * self.config.similarity_perturbation
            queries.append(base_query + noise)
        
        # clustered queries
        cluster_indices = set()
        num_centers = min(self.config.num_clusters, self.config.num_cache_queries)
        while len(cluster_indices) < num_centers:
            center_idx = np.random.randint(0, self.config.num_cache_queries)
            cluster_indices.add(center_idx)

        # generate queries for each center
        queries_per_cluster = num_clustered // len(cluster_indices)
        for center_idx in cluster_indices:
            center = self.cache_queries[center_idx]
            for _ in range(queries_per_cluster):
                noise = np.random.randn(self.config.dimension) * (self.config.similarity_perturbation * 2)
                queries.append(center + noise)
        
        # random queries
        for _ in range(num_random):
            query = np.random.randn(self.config.dimension)
            queries.append(query)
        
        # shuffle to mix them up
        queries = np.array(queries, dtype=np.float32)
        np.random.shuffle(queries)
        
        return queries
    
    def save(self, output_dir: str):
        """
        Save generated dataset to disk.
        
        Args:
            output_dir: Directory to save dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_path = output_path / self.config.name
        dataset_path.mkdir(exist_ok=True)
        
        print(f"\nSaving dataset to: {dataset_path}")
        
        # save vectors
        np.save(dataset_path / "base_vectors.npy", self.base_vectors)
        np.save(dataset_path / "cache_queries.npy", self.cache_queries)
        np.save(dataset_path / "test_queries.npy", self.test_queries)
        
        # save configuration
        with open(dataset_path / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"  Saved base_vectors.npy: {self.base_vectors.shape}")
        print(f"  Saved cache_queries.npy: {self.cache_queries.shape}")
        print(f"  Saved test_queries.npy: {self.test_queries.shape}")
        print(f"  Saved config.json")

#-------------------------------------------------------------------------------

def load_synthetic_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a previously generated synthetic dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (base_vectors, cache_queries, test_queries, config)
    """
    path = Path(dataset_path)
    
    base_vectors = np.load(path / "base_vectors.npy")
    cache_queries = np.load(path / "cache_queries.npy")
    test_queries = np.load(path / "test_queries.npy")
    
    with open(path / "config.json", 'r') as f:
        config = json.load(f)
    
    return base_vectors, cache_queries, test_queries, config

#-------------------------------------------------------------------------------

def create_standard_datasets():
    """Create standard test datasets for lemma verification."""
    
    datasets = [
        # small random queries
        DatasetConfig(
            name="small_random",
            num_base_vectors=1000,
            num_cache_queries=50,
            num_test_queries=10,
            dimension=64,
            K=10,
            N=10,
            test_query_strategy="random",
            seed=42
        ),
        
        # small similar queries
        DatasetConfig(
            name="small_similar",
            num_base_vectors=1000,
            num_cache_queries=50,
            num_test_queries=10,
            dimension=128,
            K=10,
            N=10,
            test_query_strategy="similar",
            similarity_perturbation=0.001,
            seed=42
        ),
        
        # small clustered queries
        DatasetConfig(
            name="small_clustered",
            num_base_vectors=1000,
            num_cache_queries=50,
            num_test_queries=20,
            dimension=128,
            K=10,
            N=10,
            test_query_strategy="clustered",
            num_clusters=10,
            similarity_perturbation=0.005,
            seed=42
        ),
        
        # mixed workload
        DatasetConfig(
            name="small_mixed",
            num_base_vectors=1000,
            num_cache_queries=50,
            num_test_queries=20,
            dimension=128,
            K=10,
            N=10,
            test_query_strategy="mixed",
            similarity_perturbation=0.001,
            seed=42
        ),
    ]
    
    output_dir = "./datasets/synthetic/data"
    
    for config in datasets:
        generator = SyntheticDatasetGenerator(config)
        generator.generate()
        generator.save(output_dir)
        print()

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    create_standard_datasets()
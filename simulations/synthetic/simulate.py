#!/usr/bin/env python3
"""
Simulation Runner for Cache Lemma Verification

Runs cache algorithms on synthetic datasets and tracks detailed metrics.
"""
#-------------------------------------------------------------------------------

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import sys

# add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.base import (
    MainMemory,
    KVCache,
    lemma1_circular_inclusion,
    lemma2_half_gap,
    combined_algorithm,
    verify_cache_hit,
    DistanceMetric,
    get_distance_function
)
from datasets.synthetic.generator import load_synthetic_dataset

#-------------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Results from a single query."""
    query_id: int
    algorithm: str
    cache_hit: bool
    correct: bool
    distance_calculations: int
    time_us: float
    hit_source: Optional[str] = None
    vectors_from_cache: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)

#-------------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregate simulation results."""
    dataset_name: str
    algorithm: str
    metric: str
    total_queries: int
    cache_hits: int
    correct_results: int
    total_distance_calcs: int
    total_time_us: float
    lemma1_hits: int = 0
    lemma2_hits: int = 0
    
    query_results: List[QueryResult] = None
    
    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct_results / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def avg_distance_calcs(self) -> float:
        return self.total_distance_calcs / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def avg_time_us(self) -> float:
        return self.total_time_us / self.total_queries if self.total_queries > 0 else 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['hit_rate'] = self.hit_rate
        d['accuracy'] = self.accuracy
        d['avg_distance_calcs'] = self.avg_distance_calcs
        d['avg_time_us'] = self.avg_time_us
        return d

#-------------------------------------------------------------------------------

class DistanceCalculationTracker:
    """Wrapper to count distance calculations."""
    
    def __init__(self, distance_func):
        self.distance_func = distance_func
        self.count = 0
    
    def __call__(self, a, b):
        self.count += 1
        return self.distance_func(a, b)
    
    def reset(self):
        self.count = 0

#-------------------------------------------------------------------------------

class CacheSimulator:
    """Simulate cache algorithms on synthetic datasets."""
    
    def __init__(
        self,
        base_vectors: np.ndarray,
        cache_queries: np.ndarray,
        test_queries: np.ndarray,
        K: int,
        N: int,
        metric: DistanceMetric = "euclidean"
    ):
        """
        Initialize simulator.
        
        Args:
            base_vectors: Database vectors
            cache_queries: Queries to populate cache
            test_queries: Queries to test
            K: Number of vectors to cache per query
            N: Number of vectors to retrieve per query
            metric: Distance metric to use
        """
        self.base_vectors = base_vectors
        self.cache_queries = cache_queries
        self.test_queries = test_queries
        self.K = K
        self.N = N
        self.metric = metric
        
        # create main memory from base vectors
        self.main_memory = MainMemory(len(base_vectors), base_vectors.shape[1])
        self.main_memory.vectors = list(base_vectors)
        
        # setup distance tracking
        self.distance_func = get_distance_function(metric)
        self.distance_tracker = DistanceCalculationTracker(self.distance_func)
        
        # initialize cache
        self.cache = None
    
    def populate_cache(self):
        """Populate cache with cache_queries."""
        print(f"Populating cache with {len(self.cache_queries)} queries...")
        self.cache = KVCache(metric=self.metric)
        
        for query in self.cache_queries:
            # compute top-K for this query
            top_k_vecs, top_k_dists, gap = self.main_memory.top_k_search(
                query, self.K, self.metric
            )
            self.cache.add_entry(query, top_k_vecs, top_k_dists, gap)
        
        print(f"Cache populated with {self.cache.size()} entries")
    
    def run_query(
        self,
        query_id: int,
        query: np.ndarray,
        algorithm: str
    ) -> QueryResult:
        """
        Run a single query through the cache algorithm.
        
        Args:
            query_id: Query identifier
            query: Query vector
            algorithm: Algorithm to use ("lemma1", "lemma2", "combined", "brute")
            
        Returns:
            QueryResult with metrics
        """
        self.distance_tracker.reset()
        start_time = time.perf_counter()
        
        if algorithm == "brute":
            # Brute force: compute distance to all vectors
            result, is_hit, metadata = None, False, {}
            for vec in self.base_vectors:
                _ = self.distance_tracker(query, vec)
        elif algorithm == "lemma1":
            result, is_hit, metadata = lemma1_circular_inclusion(
                query, self.cache.get_all_entries(), self.N, self.metric
            )
        elif algorithm == "lemma2":
            result, is_hit, metadata = lemma2_half_gap(
                query, self.cache.get_all_entries(), self.N, self.metric
            )
        elif algorithm == "combined":
            result, is_hit, metadata = combined_algorithm(
                query, self.cache.get_all_entries(), self.N, self.metric
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        elapsed_us = (time.perf_counter() - start_time) * 1e6
        
        # Verify correctness if we got a cache hit
        is_correct = False
        if is_hit and result is not None:
            is_correct, _ = verify_cache_hit(
                query, result, self.N, self.main_memory, self.metric
            )
        
        return QueryResult(
            query_id=query_id,
            algorithm=algorithm,
            cache_hit=is_hit,
            correct=is_correct,
            distance_calculations=self.distance_tracker.count,
            time_us=elapsed_us,
            hit_source=metadata.get('hit_lemma') or metadata.get('hit_source'),
            vectors_from_cache=metadata.get('vectors_added', 0)
        )
    
    def run_simulation(
        self,
        algorithm: str,
        verbose: bool = False
    ) -> SimulationResult:
        """
        Run complete simulation with all test queries.
        
        Args:
            algorithm: Algorithm to use
            verbose: Print progress
            
        Returns:
            SimulationResult with aggregate metrics
        """
        print(f"\n{'='*80}")
        print(f"Running simulation: {algorithm} on {self.metric} distance")
        print(f"{'='*80}")
        
        if algorithm != "brute":
            self.populate_cache()
        
        query_results = []
        cache_hits = 0
        correct_results = 0
        total_distance_calcs = 0
        total_time_us = 0.0
        lemma1_hits = 0
        lemma2_hits = 0
        
        for i, query in enumerate(self.test_queries):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.test_queries)} queries...")
            
            result = self.run_query(i, query, algorithm)
            query_results.append(result)
            
            if result.cache_hit:
                cache_hits += 1
            if result.correct:
                correct_results += 1
            
            total_distance_calcs += result.distance_calculations
            total_time_us += result.time_us
            
            if result.hit_source == "lemma1":
                lemma1_hits += 1
            elif result.hit_source == "lemma2":
                lemma2_hits += 1
        
        sim_result = SimulationResult(
            dataset_name="",
            algorithm=algorithm,
            metric=self.metric,
            total_queries=len(self.test_queries),
            cache_hits=cache_hits,
            correct_results=correct_results,
            total_distance_calcs=total_distance_calcs,
            total_time_us=total_time_us,
            lemma1_hits=lemma1_hits,
            lemma2_hits=lemma2_hits,
            query_results=query_results
        )
        
        self._print_summary(sim_result)
        
        return sim_result
    
    def _print_summary(self, result: SimulationResult):
        """Print simulation summary."""
        print(f"\n{'='*80}")
        print("SIMULATION SUMMARY")
        print(f"{'='*80}")
        print(f"Algorithm: {result.algorithm}")
        print(f"Metric: {result.metric}")
        print(f"Total Queries: {result.total_queries}")
        print(f"Cache Hits: {result.cache_hits} ({result.hit_rate:.1%})")
        print(f"  - Lemma 1 Hits: {result.lemma1_hits}")
        print(f"  - Lemma 2 Hits: {result.lemma2_hits}")
        print(f"Correct Results: {result.correct_results} ({result.accuracy:.1%})")
        print(f"Avg Distance Calcs: {result.avg_distance_calcs:.1f}")
        print(f"Avg Time: {result.avg_time_us:.1f} μs")
        print(f"{'='*80}\n")

#-------------------------------------------------------------------------------

def run_dataset_simulations(
    dataset_path: str,
    algorithms: List[str] = None,
    metrics: List[DistanceMetric] = None,
    output_dir: str = "./simulations/synthetic/raw"
):
    """
    Run simulations on a dataset.
    
    Args:
        dataset_path: Path to synthetic dataset
        algorithms: List of algorithms to test
        metrics: List of metrics to test
        output_dir: Directory to save results
    """
    if algorithms is None:
        algorithms = ["lemma1", "lemma2", "combined"]
    
    if metrics is None:
        metrics = ["euclidean", "angular"]
    
    print(f"Loading dataset from: {dataset_path}")
    base_vectors, cache_queries, test_queries, config = load_synthetic_dataset(dataset_path)
    dataset_name = Path(dataset_path).name
    
    print(f"Dataset: {dataset_name}")
    print(f"  Base vectors: {base_vectors.shape}")
    print(f"  Cache queries: {cache_queries.shape}")
    print(f"  Test queries: {test_queries.shape}")
    print(f"  K={config['K']}, N={config['N']}")
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for metric in metrics:
        for algorithm in algorithms:
            simulator = CacheSimulator(
                base_vectors, cache_queries, test_queries,
                config['K'], config['N'], metric
            )
            
            result = simulator.run_simulation(algorithm, verbose=True)
            result.dataset_name = dataset_name
            all_results.append(result)
            
            result_file = output_path / f"{algorithm}_{metric}.json"
            with open(result_file, 'w') as f:
                result_dict = result.to_dict()
                result_dict.pop('query_results', None)
                json.dump(result_dict, f, indent=2)
            
            print(f"Saved results to: {result_file}")
    
    summary = {
        "dataset_name": dataset_name,
        "config": config,
        "results": [r.to_dict() for r in all_results]
    }
    
    for r in summary["results"]:
        r.pop('query_results', None)
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved summary to: {summary_file}")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    datasets_dir = Path("./datasets/synthetic/data")
    
    if not datasets_dir.exists():
        print(f"Error: Datasets directory not found: {datasets_dir}")
        print("Please run: make generate-datasets first")
        sys.exit(1)
    
    datasets = [d for d in datasets_dir.iterdir() if d.is_dir()]
    
    if not datasets:
        print(f"No datasets found in: {datasets_dir}")
        print("Please run: make generate-datasets first")
        sys.exit(1)
    
    print(f"Found {len(datasets)} datasets")
    
    for dataset_path in datasets:
        run_dataset_simulations(
            str(dataset_path),
            algorithms=["lemma1", "lemma2", "combined"],
            metrics=["euclidean", "angular"]
        )
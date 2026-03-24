#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.base import (
    MainMemory,
    KVCache,
    lemma1_circular_inclusion,
    lemma2_half_gap,
    combined_algorithm,
    lemma1_no_union,
    lemma2_no_union,
    combined_no_union,
    DistanceMetric,
    get_distance_function
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Results from a single query."""
    query_id: int
    algorithm: str
    cache_hit: bool
    correct: bool
    distance_calculations: Optional[int]  # calls to distance function during cache lookup only;
                                           # excludes: cache population, ground truth precomputation,
                                           # and final result sorting in union algorithm variants;
                                           # None for brute force (not a valid comparison)
    time_us: Optional[float]              # microseconds for algorithm execution only, None for brute force
    hit_source: Optional[str] = None
    vectors_from_cache: int = 0
    error: Optional[str] = None
    result_indices: Optional[List[int]] = None  # Indices of result vectors for validation
    
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
    total_distance_calcs: Optional[int]   # None for brute force
    total_time_us: Optional[float]        # None for brute force
    lemma1_hits: int = 0
    lemma2_hits: int = 0
    
    query_results: List[QueryResult] = None
    
    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Accuracy of cache hits: correct_results / cache_hits."""
        return self.correct_results / self.cache_hits if self.cache_hits > 0 else 1
    
    @property
    def avg_distance_calcs(self) -> Optional[float]:
        # None for brute force (not tracked)
        if self.total_distance_calcs is None:
            return None
        return self.total_distance_calcs / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def avg_time_us(self) -> Optional[float]:
        # None for brute force (not tracked)
        if self.total_time_us is None:
            return None
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
        self.main_memory = MainMemory(vectors=list(base_vectors))
        
        # setup distance tracking
        self.distance_func = get_distance_function(metric)
        self.distance_tracker = DistanceCalculationTracker(self.distance_func)
        
        # initialize cache
        self.cache = None
    
    def populate_cache(self):
        """Populate cache with cache_queries."""
        logger.info(f"Populating cache with {len(self.cache_queries)} queries...")
        self.cache = KVCache(metric=self.metric)
        
        for idx, query in enumerate(self.cache_queries):
            # compute top-K for this query
            top_k_vecs, top_k_dists, gap = self.main_memory.top_k_search(
                query, self.K, self.metric
            )
            self.cache.add_entry(query, top_k_vecs, top_k_dists, gap)
            
            if idx < 3:
                logger.debug(f"Cache entry {idx}: K={len(top_k_vecs)}, gap={gap:.6f}, "
                           f"first_dist={top_k_dists[0]:.6f}, last_dist={top_k_dists[-1]:.6f}")
        
        logger.info(f"Cache populated with {self.cache.size()} entries")
    
    def run_query(
        self,
        query_id: int,
        query: np.ndarray,
        algorithm: str,
        ground_truth_indices: Optional[List[int]] = None,
        override_N: Optional[int] = None
    ) -> QueryResult:
        """
        Run a single query through the cache algorithm.
        
        Args:
            query_id: Query identifier
            query: Query vector
            algorithm: Algorithm to use ("lemma1", "lemma2", "combined", "brute")
            ground_truth_indices: Optional precomputed ground truth indices for verification
            
        Returns:
            QueryResult with metrics
        """
        self.distance_tracker.reset()
        # time_us covers algorithm execution only; index lookup and validation happen after elapsed_us is recorded
        start_time = time.perf_counter()

        if query_id < 3:
            logger.debug(f"Query {query_id} ({algorithm}): query_shape={query.shape}")
            if ground_truth_indices is not None:
                logger.debug(f"  Ground truth: {len(ground_truth_indices)} indices")
        
        N = override_N if override_N is not None else self.N
        
        metadata = {}
        try:
            if algorithm == "brute":
                # brute force: compute actual top-N from main memory;
                # performance (distance_calculations, time_us) is not tracked -- brute force
                # touches all M base vectors in-memory, which doesn't reflect real memory access costs
                result, _, _ = self.main_memory.top_k_search(query, N, self.metric)
                is_hit = False
            elif algorithm == "lemma1":
                result, is_hit, metadata = lemma1_circular_inclusion(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            elif algorithm == "lemma1_no_union":
                result, is_hit, metadata = lemma1_no_union(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            elif algorithm == "lemma2":
                result, is_hit, metadata = lemma2_half_gap(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            elif algorithm == "lemma2_no_union":
                result, is_hit, metadata = lemma2_no_union(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            elif algorithm == "combined":
                result, is_hit, metadata = combined_algorithm(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            elif algorithm == "combined_no_union":
                result, is_hit, metadata = combined_no_union(
                    query, self.cache.get_all_entries(), N, self.metric,
                    distance_tracker=self.distance_tracker
                )
            else:
                raise ValueError(f"u dumb: {algorithm}")
        except Exception as e:
            # capture any error that occurred while attempting to use the cache
            logger.exception("Error while running algorithm %s for query %s", algorithm, query_id)
            result = None
            is_hit = False
            metadata = { 'error': str(e) }
        
        elapsed_us = None if algorithm == "brute" else (time.perf_counter() - start_time) * 1e6

        # convert result vectors to indices for efficient validation
        result_indices = None
        if result is not None:
            result_indices = []
            for vec in result:
                # find index in base_vectors
                matches = np.where((self.base_vectors == vec).all(axis=1))[0]
                if len(matches) > 0:
                    result_indices.append(int(matches[0]))
        
        # determine correctness
        is_correct = False
        
        if algorithm == "brute":
            # brute force is always "correct" by definition, compare if possible
            if result is not None and ground_truth_indices is not None and result_indices is not None:
                is_correct = self._compare_results_by_indices(result_indices, ground_truth_indices)
            elif result is not None:
                is_correct = True  # assume brute force is correct
        elif is_hit and result is not None:
            # for cache algorithms, verify if hit was correct
            if ground_truth_indices is not None and result_indices is not None:
                is_correct = self._compare_results_by_indices(result_indices, ground_truth_indices)
            else:
                is_correct = False  # no ground truth provided; treat as unverified
        
        if query_id < 3:
            logger.debug(f"  Result: hit={is_hit}, correct={is_correct}, "
                       f"result_len={len(result) if result else 0}, "
                       f"dist_calcs={self.distance_tracker.count}")
        
        return QueryResult(
            query_id=query_id,
            algorithm=algorithm,
            cache_hit=is_hit,
            correct=is_correct,
            distance_calculations=None if algorithm == "brute" else self.distance_tracker.count,
            time_us=elapsed_us,
            hit_source=metadata.get('hit_lemma') or metadata.get('hit_source'),
            vectors_from_cache=metadata.get('vectors_added', 0),
            error=metadata.get('error'),
            result_indices=result_indices
        )
    
    def _compare_results_by_indices(self, result_indices: List[int], ground_truth_indices: List[int]) -> bool:
        """
        Compare two result sets by indices.
        
        Args:
            result_indices: Result vector indices
            ground_truth_indices: Ground truth vector indices
            
        Returns:
            True if results match the ground truth indices, False otherwise
        """
        if len(result_indices) != len(ground_truth_indices):
            logger.debug(f"Length mismatch: result={len(result_indices)}, ground_truth={len(ground_truth_indices)}")
            return False
        
        result_set = set(result_indices)
        gt_set = set(ground_truth_indices)
        match = result_set == gt_set
        
        if not match:
            logger.debug(f"Index set mismatch: {len(result_set)} vs {len(gt_set)} unique indices")
            only_in_result = result_set - gt_set
            only_in_gt = gt_set - result_set
            if only_in_result:
                logger.debug(f"Only in result: {sorted(only_in_result)[:5]}")
            if only_in_gt:
                logger.debug(f"Only in GT: {sorted(only_in_gt)[:5]}")
        
        return match
    
    def run_simulation(
        self,
        algorithm: str,
        verbose: bool = False,
        populate: bool = True
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
        
        if algorithm != "brute" and populate:
            self.populate_cache()
        
        query_results = []
        cache_hits = 0
        correct_results = 0
        total_distance_calcs = None if algorithm == "brute" else 0
        total_time_us = None if algorithm == "brute" else 0.0
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

            if result.distance_calculations is not None:
                total_distance_calcs += result.distance_calculations
            if result.time_us is not None:
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
        dc = f"{result.avg_distance_calcs:.1f}" if result.avg_distance_calcs is not None else "N/A (brute)"
        t = f"{result.avg_time_us:.1f} us" if result.avg_time_us is not None else "N/A (brute)"
        print(f"Avg Distance Calcs: {dc}")
        print(f"Avg Time: {t}")
        print(f"{'='*80}\n")

    def cross_validate_angular_vs_cosine(
        self,
        angular_result: SimulationResult,
        cosine_brute_result: SimulationResult
    ) -> Dict:
        """
        Cross-validate angular algorithm results against cosine brute force.
        Angular and cosine should produce identical result sets.
        
        Args:
            angular_result: Results from running algorithm with angular distance
            cosine_brute_result: Results from brute force with cosine distance
            
        Returns:
            Dict with validation statistics
        """
        if cosine_brute_result.algorithm != "brute":
            raise ValueError("cosine_result must be from brute force algorithm")
        
        if angular_result.metric != "angular":
            raise ValueError("angular_result must use angular metric")
        
        if cosine_brute_result.metric != "cosine":
            raise ValueError("cosine_result must use cosine metric")
        
        print(f"\n{'='*80}")
        print(f"Comparing {angular_result.algorithm} against cosine")
        
        total_queries = len(angular_result.query_results)
        matching_results = 0
        mismatches = []
        
        for i in range(total_queries):
            angular_qr = angular_result.query_results[i]
            cosine_qr = cosine_brute_result.query_results[i]
            
            if not angular_qr.cache_hit:
                continue
            
            # use pre-computed result indices
            if angular_qr.result_indices is None or cosine_qr.result_indices is None:
                print(f"Warning: Query {i} missing result indices, skipping")
                continue
            
            # compare sets of indices
            angular_set = set(angular_qr.result_indices)
            cosine_set = set(cosine_qr.result_indices)
            
            if angular_set == cosine_set:
                matching_results += 1
            else:
                mismatches.append({
                    'query_id': i,
                    'angular_only': len(angular_set - cosine_set),
                    'cosine_only': len(cosine_set - angular_set),
                    'angular_indices': sorted(angular_set - cosine_set)[:5], # store 5 for obs
                    'cosine_indices': sorted(cosine_set - angular_set)[:5]
                })
        
        cache_hits = sum(1 for qr in angular_result.query_results if qr.cache_hit)
        match_rate = matching_results / cache_hits if cache_hits > 0 else 0.0
        
        validation_result = {
            'total_queries': total_queries,
            'cache_hits_checked': cache_hits,
            'matching_results': matching_results,
            'mismatch_count': len(mismatches),
            'match_rate': match_rate,
            'mismatches': mismatches[:10] # first 10 for obs
        }
        
        print(f"Total queries: {total_queries}")
        print(f"Cache hits checked: {cache_hits}")
        print(f"Matching result sets: {matching_results}/{cache_hits} ({match_rate:.1%})")

        if mismatches:
            print(f"Mismatches found, examine this case: {angular_result.query_results[mismatches[0]['query_id']]}")
        
        return validation_result

#-------------------------------------------------------------------------------
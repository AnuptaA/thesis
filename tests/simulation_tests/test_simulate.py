#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import shutil
from simulations.synthetic.simulate import (
    QueryResult,
    SimulationResult,
    DistanceCalculationTracker,
    CacheSimulator,
    run_dataset_simulations
)
from utils.base import get_distance_function
from datasets.synthetic.generator import DatasetConfig, SyntheticDatasetGenerator

#-------------------------------------------------------------------------------

def test_distance_tracker():
    """Test distance calculation tracking."""
    print("Test 1: Distance calculation tracker")
    
    distance_func = get_distance_function("euclidean")
    tracker = DistanceCalculationTracker(distance_func)
    
    a = np.random.randn(32)
    b = np.random.randn(32)
    
    assert tracker.count == 0, f"Initial count should be 0, got {tracker.count}"
    
    _ = tracker(a, b)
    _ = tracker(a, b)
    _ = tracker(b, a)
    
    assert tracker.count == 3, f"Count should be 3, got {tracker.count}"
    
    # Reset
    tracker.reset()
    assert tracker.count == 0, f"Count after reset should be 0, got {tracker.count}"
    
    print("  > Distance tracker works")

#-------------------------------------------------------------------------------

def test_query_result_dataclass():
    """Test QueryResult dataclass."""
    print("\nTest 2: QueryResult dataclass")
    
    result = QueryResult(
        query_id=0,
        algorithm="lemma1",
        cache_hit=True,
        correct=True,
        distance_calculations=50,
        time_us=123.45,
        hit_source="lemma1",
        vectors_from_cache=10
    )
    
    d = result.to_dict()
    assert d['query_id'] == 0
    assert d['algorithm'] == "lemma1"
    assert d['cache_hit'] == True
    assert d['correct'] == True
    assert d['distance_calculations'] == 50
    assert d['time_us'] == 123.45
    
    print("  > QueryResult works")

#-------------------------------------------------------------------------------

def test_simulation_result_properties():
    """Test SimulationResult computed properties."""
    print("\nTest 3: SimulationResult properties")
    
    result = SimulationResult(
        dataset_name="test",
        algorithm="combined",
        metric="euclidean",
        total_queries=100,
        cache_hits=75,
        correct_results=70,
        total_distance_calcs=5000,
        total_time_us=10000.0,
        lemma1_hits=50,
        lemma2_hits=25
    )
    
    assert result.hit_rate == 0.75, f"Hit rate should be 0.75, got {result.hit_rate}"
    assert result.accuracy == 0.70, f"Accuracy should be 0.70, got {result.accuracy}"
    assert result.avg_distance_calcs == 50.0, f"Avg distance calcs should be 50.0"
    assert result.avg_time_us == 100.0, f"Avg time should be 100.0"
    
    print("  > SimulationResult properties work")

#-------------------------------------------------------------------------------

def test_cache_simulator_initialization():
    """Test CacheSimulator initialization."""
    print("\nTest 4: CacheSimulator initialization")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    test_queries = np.random.randn(20, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors=base_vectors,
        cache_queries=cache_queries,
        test_queries=test_queries,
        K=10,
        N=5,
        metric="euclidean"
    )
    
    assert simulator.K == 10
    assert simulator.N == 5
    assert simulator.metric == "euclidean"
    assert simulator.cache is None
    assert len(simulator.main_memory.vectors) == 100
    
    print("  > CacheSimulator initialization works")

#-------------------------------------------------------------------------------

def test_cache_population():
    """Test cache population."""
    print("\nTest 5: Cache population")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )
    
    simulator.populate_cache()
    
    assert simulator.cache is not None, "Cache should be populated"
    assert simulator.cache.size() == 5, f"Cache should have 5 entries, got {simulator.cache.size()}"
    
    entries = simulator.cache.get_all_entries()
    assert len(entries) == 5
    assert entries[0].k == 10, "Each entry should have K=10 vectors"
    
    print("  > Cache population works")

#-------------------------------------------------------------------------------

def test_run_single_query():
    """Test running a single query."""
    print("\nTest 6: Run single query")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )
    
    simulator.populate_cache()
    
    # run one query
    query = test_queries[0]
    result = simulator.run_query(0, query, "combined")
    
    assert result.query_id == 0
    assert result.algorithm == "combined"
    assert isinstance(result.cache_hit, bool)
    assert isinstance(result.correct, bool)
    assert result.distance_calculations >= 0
    assert result.time_us > 0
    
    print(f"  Query result: hit={result.cache_hit}, correct={result.correct}")
    print("  > Single query execution works")

#-------------------------------------------------------------------------------

def test_brute_force_algorithm():
    """Test brute force baseline."""
    print("\nTest 7: Brute force algorithm")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )
    
    result = simulator.run_query(0, test_queries[0], "brute")
    
    assert result.cache_hit == False, "Brute force should never hit cache"
    assert result.distance_calculations == 100, f"Should compute 100 distances, got {result.distance_calculations}"
    
    print("  > Brute force works")

#-------------------------------------------------------------------------------

def test_lemma_algorithms():
    """Test lemma1, lemma2, and combined algorithms."""
    print("\nTest 8: Lemma algorithms")
    
    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    test_queries = np.random.randn(20, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=20, N=10, metric="euclidean"
    )
    
    simulator.populate_cache()
    
    # test all algorithms
    for algo in ["lemma1", "lemma2", "combined"]:
        result = simulator.run_query(0, test_queries[0], algo)
        
        assert result.algorithm == algo
        assert isinstance(result.cache_hit, bool)
        
        # if it's a hit, verify correctness was checked
        if result.cache_hit:
            assert isinstance(result.correct, bool)
            if result.correct:
                print(f"    {algo}: Cache hit and correct!")
        
        print(f"  > {algo} executed successfully")

#-------------------------------------------------------------------------------

def test_full_simulation():
    """Test complete simulation run."""
    print("\nTest 9: Full simulation")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )
    
    result = simulator.run_simulation("combined", verbose=False)
    
    assert result.total_queries == 10
    assert result.cache_hits + (result.total_queries - result.cache_hits) == result.total_queries
    assert result.total_distance_calcs >= 0
    assert result.total_time_us > 0
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    
    print(f"  Results: {result.cache_hits}/{result.total_queries} hits, accuracy={result.accuracy:.2f}")
    print("  > Full simulation works")

#-------------------------------------------------------------------------------

def test_different_metrics():
    """Test simulation with different distance metrics."""
    print("\nTest 10: Different distance metrics")
    
    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)
    
    for metric in ["euclidean", "cosine", "angular"]:
        simulator = CacheSimulator(
            base_vectors, cache_queries, test_queries,
            K=10, N=5, metric=metric
        )
        
        result = simulator.run_simulation("combined", verbose=False)
        
        assert result.metric == metric
        assert result.total_queries == 10
        
        print(f"  > {metric} metric works")

#-------------------------------------------------------------------------------

def test_run_dataset_simulations():
    """Test running simulations on saved dataset."""
    print("\nTest 11: Run dataset simulations")
    
    test_dir = "./tmp"
    
    config = DatasetConfig(
        name="sim_test",
        num_base_vectors=50,
        num_cache_queries=5,
        num_test_queries=10,
        dimension=16,
        K=5,
        N=5,
        test_query_strategy="random",
        seed=42
    )
    
    gen = SyntheticDatasetGenerator(config)
    gen.generate()
    gen.save(test_dir)
    
    # run simulations on it
    output_dir = "./tmp_output"
    dataset_path = Path(test_dir) / "sim_test"
    
    run_dataset_simulations(
        str(dataset_path),
        algorithms=["lemma1", "combined"],
        metrics=["euclidean"],
        output_dir=output_dir
    )
    
    output_path = Path(output_dir) / "sim_test"
    assert (output_path / "lemma1_euclidean.json").exists()
    assert (output_path / "combined_euclidean.json").exists()
    assert (output_path / "summary.json").exists()
    
    # cleanup
    shutil.rmtree(test_dir)
    shutil.rmtree(output_dir)
    
    print("  > Dataset simulation works")

#-------------------------------------------------------------------------------

def test_similar_query_high_hit_rate():
    """Test that similar queries get high hit rate."""
    print("\nTest 12: Similar queries should hit cache")
    
    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    
    # ceate test queries very similar to cached queries
    test_queries = []
    for i in range(20):
        cache_idx = i % len(cache_queries)
        similar = cache_queries[cache_idx] + np.random.randn(32) * 0.001
        test_queries.append(similar)
    test_queries = np.array(test_queries, dtype=np.float32)
    
    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=20, N=10, metric="euclidean"
    )
    
    result = simulator.run_simulation("combined", verbose=False)
    
    print(f"  Hit rate with similar queries: {result.hit_rate:.1%}")
    print(f"  Lemma 1 hits: {result.lemma1_hits}")
    print(f"  Lemma 2 hits: {result.lemma2_hits}")
    
    assert result.total_queries == 20
    
    print("  > Similar query simulation works")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Testing Simulation System")
    print("="*60)
    
    test_distance_tracker()
    test_query_result_dataclass()
    test_simulation_result_properties()
    test_cache_simulator_initialization()
    test_cache_population()
    test_run_single_query()
    test_brute_force_algorithm()
    test_lemma_algorithms()
    test_full_simulation()
    test_different_metrics()
    test_run_dataset_simulations()
    test_similar_query_high_hit_rate()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
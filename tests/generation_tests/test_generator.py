#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import shutil
from datasets.synthetic.generator import (
    DatasetConfig,
    SyntheticDatasetGenerator,
    load_synthetic_dataset
)

#-------------------------------------------------------------------------------

def test_random_generation():
    """Test random query generation."""
    print("Test 1: Random query generation")
    
    config = DatasetConfig(
        name="test_random",
        num_base_vectors=100,
        num_cache_queries=10,
        num_test_queries=20,
        dimension=32,
        K=5,
        N=5,
        test_query_strategy="random",
        seed=42
    )
    
    gen = SyntheticDatasetGenerator(config)
    base, cache_q, test_q = gen.generate()
    
    assert base.shape == (100, 32), f"Wrong base shape: {base.shape}"
    assert cache_q.shape == (10, 32), f"Wrong cache shape: {cache_q.shape}"
    assert test_q.shape == (20, 32), f"Wrong test shape: {test_q.shape}"
    
    print("  > Random generation works")

#-------------------------------------------------------------------------------

def test_similar_generation():
    """Test similar query generation."""
    print("\nTest 2: Similar query generation")
    
    config = DatasetConfig(
        name="test_similar",
        num_base_vectors=100,
        num_cache_queries=10,
        num_test_queries=20,
        dimension=32,
        K=5,
        N=5,
        test_query_strategy="similar",
        similarity_perturbation=0.01,
        seed=42
    )
    
    gen = SyntheticDatasetGenerator(config)
    _, cache_q, test_q = gen.generate()
    
    # check that SOME test queries are close to SOME cache queries
    min_distances = []
    for test_vec in test_q[:10]:  # check first 10 test queries
        dists = [np.linalg.norm(test_vec - cache_vec) for cache_vec in cache_q]
        min_distances.append(min(dists))
    
    # most should be very close (within perturbation range)
    avg_min_dist = np.mean(min_distances)
    assert avg_min_dist < 0.5, f"Average minimum distance too large: {avg_min_dist}"

    print("  > Similar generation works")

#-------------------------------------------------------------------------------

def test_clustered_generation():
    """Test clustered query generation."""
    print("\nTest 3: Clustered query generation")
    
    config = DatasetConfig(
        name="test_clustered",
        num_base_vectors=100,
        num_cache_queries=10,
        num_test_queries=50,
        dimension=32,
        K=5,
        N=5,
        test_query_strategy="clustered",
        num_clusters=5,
        similarity_perturbation=0.01,
        seed=42
    )
    
    gen = SyntheticDatasetGenerator(config)
    _, _, test_q = gen.generate()
    
    assert test_q.shape == (50, 32), f"Wrong test shape: {test_q.shape}"
    print("  > Clustered generation works")

#-------------------------------------------------------------------------------

def test_save_and_load():
    """Test saving and loading dataset."""
    print("\nTest 4: Save and load")
    
    test_dir = "./tests/generation_tests/tmp"
    
    config = DatasetConfig(
        name="test_save",
        num_base_vectors=50,
        num_cache_queries=5,
        num_test_queries=10,
        dimension=16,
        K=3,
        N=3,
        test_query_strategy="random",
        seed=42
    )
    
    # generate and save
    gen = SyntheticDatasetGenerator(config)
    base_orig, cache_orig, test_orig = gen.generate()
    gen.save(test_dir)
    
    # load back
    dataset_path = Path(test_dir) / "test_save"
    base_loaded, cache_loaded, test_loaded, _ = load_synthetic_dataset(str(dataset_path))
    
    # check shapes match
    assert base_loaded.shape == base_orig.shape, "Base vectors don't match"
    assert cache_loaded.shape == cache_orig.shape, "Cache queries don't match"
    assert test_loaded.shape == test_orig.shape, "Test queries don't match"

    # check values match
    assert np.allclose(base_loaded, base_orig), "Base vector values don't match"
    assert np.allclose(cache_loaded, cache_orig), "Cache query values don't match"
    
    # cleanup
    shutil.rmtree(test_dir)
    
    print("  > Save and load works")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*60)
    print("Testing Dataset Generation")
    print("="*60)
    
    test_random_generation()
    test_similar_generation()
    test_clustered_generation()
    test_save_and_load()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)

#-------------------------------------------------------------------------------
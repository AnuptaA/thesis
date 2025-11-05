#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import struct

#-------------------------------------------------------------------------------

def load_fvecs(filepath: str) -> np.ndarray:
    """
    Load fvecs file and return as numpy array.
    
    Args:
        filepath: Path to fvecs file

    Returns:
        Numpy array of shape (n_vectors, vector_dim)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    dim = struct.unpack('i', data[:4])[0]
    bytes_per_vector = 4 * (dim + 1)
    n_vectors = len(data) // bytes_per_vector

    vectors = np.empty((n_vectors, dim), dtype=np.float32)
    offset = 0

    for i in range(n_vectors):
        offset += 4 
        vector = struct.unpack('f' * dim, data[offset:offset + 4 * dim])
        vectors[i] = np.array(vector, dtype=np.float32)
        offset += 4 * dim

    return vectors

#-------------------------------------------------------------------------------

def load_ivecs(filepath: str) -> np.ndarray:
    """
    Load ivecs file and return as numpy array.
    
    Args:
        filepath: Path to ivecs file

    Returns:
        Numpy array of shape (n_vectors, vector_dim)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    dim = struct.unpack('i', data[:4])[0]
    bytes_per_vector = 4 * (dim + 1)
    n_vectors = len(data) // bytes_per_vector

    vectors = np.empty((n_vectors, dim), dtype=np.int32)
    offset = 0

    for i in range(n_vectors):
        offset += 4
        vector = struct.unpack('i' * dim, data[offset:offset + 4 * dim])
        vectors[i] = np.array(vector, dtype=np.int32)
        offset += 4 * dim

    return vectors

#-------------------------------------------------------------------------------


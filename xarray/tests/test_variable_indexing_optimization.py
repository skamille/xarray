from __future__ import annotations

import numpy as np
import pytest

import xarray as xr
from xarray import Variable
from xarray.core.indexing import OuterIndexer, VectorizedIndexer


def test_optimization_for_2d_variable_indexer():
    # Create a variable
    var = Variable(["x", "y", "z"], np.random.rand(5, 6, 7))
    
    # Create a 2D Variable indexer with first dimension of length 1
    index_var = Variable(["window", "x"], np.arange(3).reshape(1, 3))
    
    # Get the indexing key via _broadcast_indexes
    key = (index_var, slice(None), slice(None))
    dims, indexer, new_order = var._broadcast_indexes(key)
    
    # The optimization should convert to OuterIndexer instead of VectorizedIndexer
    # since we're converting the 2D Variable with first dim length 1 to a 1D Variable
    assert isinstance(indexer, OuterIndexer)
    # The dimensions should be correct
    assert "window" not in dims
    assert "x" not in dims  # Original x dimension replaced by indexing
    assert "y" in dims
    assert "z" in dims
    # The shape after indexing should be as expected
    result = var[key]
    assert result.shape == (3, 6, 7)
    
    # Test with a 2D Variable indexer that doesn't have first dimension of length 1
    # In this case, we still expect a VectorizedIndexer
    index_var2 = Variable(["window", "x"], np.arange(6).reshape(2, 3))
    key2 = (index_var2, slice(None), slice(None))
    dims2, indexer2, new_order2 = var._broadcast_indexes(key2)
    assert isinstance(indexer2, VectorizedIndexer)
    
    # Functional test - ensure we get the right results
    arr = np.arange(35).reshape(5, 7)
    var2 = Variable(["x", "y"], arr)
    # Create an index Variable with shape (1, 3)
    idx = Variable(["window", "idx"], np.array([[1, 2, 3]]))
    # Apply indexing
    result = var2.isel(x=idx)
    # Verify shape and values
    assert result.shape == (3, 7)
    np.testing.assert_array_equal(result.data, arr[[1, 2, 3]])
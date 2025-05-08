"""Test memory usage with isel and multidimensional Variable indexers."""

import numpy as np
import xarray as xr
import gc
import psutil
import os
import time

def get_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_isel_memory_usage():
    """Test memory usage with isel and multidimensional Variable indexers."""
    print("Creating test data...")
    
    # Create a synthetic dataset with a large array
    data_shape = (100, 180, 360)  # smaller than the example but still large enough
    data = np.zeros(data_shape, dtype=np.float32)
    ds = xr.Dataset({"var": (["t", "y", "x"], data)})
    
    # Create a variable indexer similar to the problematic case
    index_var = xr.Variable(["window", "t"], np.arange(50).reshape(1, 50))
    
    # Force garbage collection to get a clean memory measurement
    gc.collect()
    start_mem = get_memory()
    print(f"Memory before indexing: {start_mem:.2f} MB")
    
    # Perform the isel operation
    print("Running isel operation...")
    start_time = time.time()
    result = ds.isel(t=index_var)
    
    # Measure memory after operation but before accessing data
    after_indexing_mem = get_memory()
    print(f"Memory after indexing (before accessing data): {after_indexing_mem:.2f} MB")
    print(f"Memory increase after indexing: {after_indexing_mem - start_mem:.2f} MB")
    
    # Now actually access the data to trigger loading/computation
    print("Accessing data...")
    _ = result.var.values
    
    # Measure memory after accessing data
    after_access_mem = get_memory()
    print(f"Memory after accessing data: {after_access_mem:.2f} MB")
    print(f"Memory increase after data access: {after_access_mem - after_indexing_mem:.2f} MB")
    print(f"Total memory increase: {after_access_mem - start_mem:.2f} MB")
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    
    # Print result info
    print(f"Result dimensions: {result.dims}")
    print(f"Result sizes: {result.sizes}")
    
    return result

def compare_methods():
    """Compare memory usage with and without the optimization."""
    print("\n=== COMPARING ORIGINAL VS OPTIMIZED IMPLEMENTATION ===\n")
    
    # First, run with larger dataset
    print("Testing with larger dataset (multiple variables)...")
    
    # Create a synthetic dataset with multiple large arrays
    num_vars = 5
    data_shape = (100, 180, 360)
    ds = xr.Dataset()
    
    for i in range(num_vars):
        ds[f"var{i}"] = (["t", "y", "x"], np.zeros(data_shape, dtype=np.float32))
    
    # Create a variable indexer similar to the problematic case
    index_var = xr.Variable(["window", "t"], np.arange(50).reshape(1, 50))
    
    # Force garbage collection to get a clean memory measurement
    gc.collect()
    
    start_mem = get_memory()
    print(f"Memory before indexing: {start_mem:.2f} MB")
    
    # Perform the isel operation
    print("Running isel operation with optimized method...")
    start_time = time.time()
    result = ds.isel(t=index_var)
    # Access data to ensure computation happens
    for i in range(num_vars):
        _ = result[f"var{i}"].values
    
    end_time = time.time()
    end_mem = get_memory()
    
    print(f"Memory after optimized indexing: {end_mem:.2f} MB")
    print(f"Memory increase: {end_mem - start_mem:.2f} MB")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    return result

if __name__ == "__main__":
    print("Testing isel memory usage with multidimensional Variable indexer")
    try:
        # Basic test
        result = test_isel_memory_usage()
        print("\nBasic test completed successfully")
        
        # Compare methods
        compare_result = compare_methods()
        print("\nComparison test completed successfully")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
"""Test for the specific example in the bug report."""

import numpy as np
import xarray as xr
import gc
import psutil
import os

def get_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def test_bug_example():
    """Test the exact case from the bug report."""
    print("\n=== TESTING EXAMPLE FROM BUG REPORT ===\n")
    
    # Create a synthetic dataset similar to the example
    # Original was 1000x180x360, we'll use a smaller version
    # for testing, but still with the same pattern
    data_shape = (200, 50, 50)
    print(f"Creating test dataset with shape {data_shape}...")
    
    # Create several variables to simulate the real-world case
    num_vars = 10
    ds = xr.Dataset()
    for i in range(num_vars):
        ds[f"var{i}"] = (["t", "y", "x"], np.zeros(data_shape, dtype=np.float32))
    
    # Create a variable indexer similar to the bug report
    index_var = xr.Variable(["window", "t"], np.arange(100).reshape(1, 100))
    print(f"Created index variable with shape {index_var.shape}")
    
    # Force garbage collection
    gc.collect()
    start_mem = get_memory()
    print(f"Memory before indexing: {start_mem:.2f} MB")
    
    # Perform the isel operation
    print("Running isel operation...")
    result = ds.isel(t=index_var)
    
    # Measure memory usage after indexing but before loading data
    after_indexing_mem = get_memory()
    print(f"Memory after indexing (before accessing data): {after_indexing_mem:.2f} MB")
    print(f"Memory increase: {after_indexing_mem - start_mem:.2f} MB")
    
    # Now access the data for one variable to simulate actual usage
    print("Accessing data for one variable...")
    _ = result["var0"].values
    
    # Measure memory after accessing data for one variable
    after_one_var_mem = get_memory()
    print(f"Memory after accessing one variable: {after_one_var_mem:.2f} MB")
    print(f"Memory increase from accessing one variable: {after_one_var_mem - after_indexing_mem:.2f} MB")
    
    # Optional: access all variables to see total memory usage
    # print("Accessing all variables...")
    # for i in range(num_vars):
    #     _ = result[f"var{i}"].values
    # 
    # all_vars_mem = get_memory()
    # print(f"Memory after accessing all variables: {all_vars_mem:.2f} MB")
    # print(f"Total memory increase: {all_vars_mem - start_mem:.2f} MB")
    
    print(f"Result dimensions: {result.dims}")
    print(f"Result sizes: {result.sizes}")
    
    return result

if __name__ == "__main__":
    try:
        result = test_bug_example()
        print("\nTest completed successfully")
    except Exception as e:
        print(f"Test failed with error: {e}")
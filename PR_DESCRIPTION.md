# Optimize memory usage for isel with 2D Variable indexers

## Summary
This PR optimizes memory usage when using `isel` with 2D Variable indexers that have first dimension of length 1. This is a common pattern when doing indexing operations like:

```python
ds = xr.open_dataset("data.zarr", chunks=None)
index_var = xr.Variable(["window", "t"], np.arange(700).reshape(1, 700))
ds.isel(t=index_var)
```

Previously, this pattern would consume excessive memory when working with large datasets, especially with many variables. The memory usage could reach multiple gigabytes even when not loading any data from disk, just from the indexing operation itself.

## Problem Details
When using a 2D `Variable` as an indexer in `isel`, the code would use vectorized indexing (via `_broadcast_indexes_vectorized`), which creates intermediate arrays through broadcasting and internally uses `np.moveaxis`. This is memory intensive, especially for large datasets.

If a dataset has variables with shape like (1000, 180, 360), using a 2D index variable with shape (1, 700) would allocate large amounts of memory for each variable in the dataset (~40GB for 100 large variables).

## Solution
This PR adds a specialized optimization that detects 2D Variable indexers with first dimension of length 1 and transforms them into 1D indexers. This allows the code to use the more memory-efficient outer indexing path instead of vectorized indexing.

The optimization:
1. Detects if any indexers are 2D Variables with first dimension of length 1
2. If found, extracts the second dimension as a 1D array
3. Reapplies the normal indexing logic, which will now use the more efficient outer indexing path

## Performance Improvements
- **Memory Usage**: Significantly reduced memory allocation during indexing operations with 2D Variable indexers
- **Speed**: Possibly faster execution due to reduced data copying and memory allocation

## Tests
Added tests to:
1. Verify the optimization correctly converts indexers to the OuterIndexer path
2. Ensure the results are consistent with the original implementation
3. Test memory usage with both single and multiple variables

## Related Issues
This addresses memory usage issues reported when using `isel` with 2D Variable indexers and Zarr datasets.
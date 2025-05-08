# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Xarray is a Python library that makes working with labeled multi-dimensional arrays simple, efficient, and fun. It introduces labels in the form of dimensions, coordinates, and attributes on top of raw NumPy-like arrays.

Core data structures:
- **Variable**: N-dimensional array with dimensions, coordinates, and attributes
- **DataArray**: Labeled N-dimensional array (extends Variable with coordinate labels)
- **Dataset**: Dictionary-like collection of DataArrays that share dimensions
- **DataTree**: Hierarchical collection of Datasets

## Development Environment Setup

### Create a development environment

```bash
# Create and activate conda environment
conda create -c conda-forge -n xarray-dev python=3.10
conda activate xarray-dev

# Install dependencies
# For Linux/MacOS
conda env update -f ci/requirements/environment.yml
# For Windows
conda env update -f ci/requirements/environment-windows.yml

# Install in development mode
pip install -e .

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Essential Commands

### Run Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest xarray/tests/test_dataarray.py
pytest xarray/tests/test_dataarray.py::TestDataArray::test_specific_method

# Run with parallel workers
pytest -n 4

# Run with specific marks
pytest -xvs -m "not slow"
```

### Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Run linting only
ruff check xarray

# Run type checking
mypy xarray
```

### Build Documentation

```bash
# Create documentation environment
conda env create -f ci/requirements/doc.yml
conda activate xarray-docs
pip install -e .

# Build docs
cd doc
make html

# Clean and rebuild
make clean
make html
```

### Performance Benchmarks

```bash
# Run benchmarks
cd asv_bench/
asv continuous -f 1.1 upstream/main HEAD

# Run specific benchmarks
asv continuous -f 1.1 upstream/main HEAD -b ^groupby
```

## Architecture Overview

### Package Structure

- **xarray/core/**: Core data structures
  - variable.py: Variable class
  - dataarray.py: DataArray class 
  - dataset.py: Dataset class
  - datatree.py: DataTree class
  - indexes.py: Indexing functionality

- **xarray/backends/**: I/O functionality
  - Support for various formats (netCDF4, zarr, h5netcdf, etc.)
  - Plugin system for third-party backends

- **xarray/computation/**: Operations on data
  - apply_ufunc.py: Universal function application
  - arithmetic.py: Arithmetic operations

- **xarray/structure/**: Combining and reshaping data
  - alignment.py: Aligns data structures
  - combine.py: Combines datasets
  - concat.py: Concatenates data

- **xarray/plot/**: Visualization functionality

### Design Patterns

1. **Immutable Data Structures**: Operations return new objects rather than modifying existing ones
2. **Lazy Evaluation**: Integration with Dask for out-of-core and parallel computing
3. **Accessor Pattern**: Namespace extension through registered accessors
4. **Dimension-aware Broadcasting**: Based on dimension names rather than array shapes

### Extension Points

1. **Accessor System**: Register custom accessors for DataArray, Dataset
2. **Backend Plugin System**: Add support for new file formats
3. **Duck Array Support**: Wrap different array implementations (NumPy, Dask, CuPy)

## Adding New Features

When adding new features:

1. Follow existing patterns in similar functionality
2. Add type annotations
3. Write comprehensive tests
4. Document in docstrings and docs/whats-new.rst
5. Consider impacts on performance and memory usage
6. Ensure backwards compatibility or provide deprecation warnings

## Conventions

1. Use dimension names consistently
2. Follow NumPy and pandas coding practices
3. Follow immutable data structure patterns
4. Use proper type annotations
5. Use ruff formatting style
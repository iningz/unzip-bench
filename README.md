# Finch.jl vs Unzipping Benchmark

Benchmark comparison of Finch.jl sparse tensor compiler vs unzipping C code for the kernel: `A(i, j) = B(i, j) * C(j, i)` (Hadamard product with transpose pattern) in CSR format.

### Project Structure

**Julia Implementation `kernels.jl`:**
-  `hadamard_transpose!(A, B, C)`

**C Implementation `kernels.c`:**
- `hadamard_transpose()`

**Benchmark `benchmark.ipynb`:**
- **Matrix Size**: Tests sizes [100, 200, 500, 1000, 2000, 3000, 5000] with fixed 5% sparsity
- **Sparsity**: Tests sparsities [1%, 2%, 5%, 10%, 15%, 20%, 30%] with fixed 1000×1000 size
- **Unzip Comparison**: Compares unzipping B vs C with varying sparsity ratios

## Setup

1. Install Julia (from https://julialang.org/downloads/)
2. Install dependencies:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## Usage

Run the correctness test:

```bash
julia --project=. test.jl
```

Run the benchmark:

Open `benchmark.ipynb` and run the cells.

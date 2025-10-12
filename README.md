# Finch.jl vs Unzipping Benchmark

Benchmark comparison of Finch.jl sparse tensor compiler vs unzipping C code for the kernel: `A(i, j) = B(i, j) * C(j, i)` (Hadamard product with transpose pattern) in CSR format.

### Project Structure

**Julia Implementation `kernels.jl`:**
-  `hadamard_transpose!(A, B, C)`

**C Implementation `kernels.c`:**
- `hadamard_transpose()`

**Benchmark `benchmark.jl`:**
- **Small**: 100×100 matrices with 10% sparsity
- **Medium**: 1000×1000 matrices with 5% sparsity
- **Large**: 5000×5000 matrices with 1% sparsity

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

```bash
julia --project=. benchmark.jl
```

Edit `benchmark.jl` `main()` function to add custom test cases:

```julia
benchmark_comparison(2000, 2000, 0.02)  # 2000×2000, 2% sparsity
```

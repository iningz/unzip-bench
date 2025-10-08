# Finch.jl vs Unzipping Benchmark

Benchmark comparison of Finch.jl sparse tensor compiler vs unzipping C code for the kernel: `A(i, j) = B(i, j) * C(j, i)` (Hadamard product with transpose pattern).

### Project Structure

**Julia Implementation:**
- `kernels.jl` - Module `FinchKernels`: kernel implementation `hadamard_transpose!(A, B, C)`
- `utils.jl` - Module `Utils`: matrix generation and Finch tensor conversion utilities
- Uses **CSR format**: `SparseList(Dense(Element(0.0)))`

**C Implementation:**
- `kernels.c` - CSR kernel: `hadamard_transpose()`
- `utils.c` - Matrix generation and memory management utilities
- `utils.h` - CSR struct definition and function declarations

**Benchmark:**
- `benchmark.jl` - Main driver using `BenchmarkTools.@benchmark` for both implementations

## Setup

1. Install Julia (from https://julialang.org/downloads/)
2. Install dependencies:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

## Usage

Run the benchmark:

```bash
julia --project=. benchmark.jl
```

The script will:
1. Compile the C library
2. Run benchmarks on three test cases:
   - **Small**: 100×100 matrices with 10% sparsity
   - **Medium**: 1000×1000 matrices with 5% sparsity
   - **Large**: 5000×5000 matrices with 1% sparsity

Edit `benchmark.jl` `main()` function to add custom test cases:

```julia
# Add custom benchmark
benchmark_comparison(2000, 2000, 0.02)  # 2000×2000, 2% sparsity
```

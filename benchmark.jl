# Benchmark Finch vs Unzipping implementation

import Finch
import SparseArrays
import BenchmarkTools
import Libdl

include("kernels.jl")
include("utils.jl")

const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function c_generate_csr_matrix(m::Int, n::Int, sparsity::Float64, seed::UInt32)
    func = Libdl.dlsym(LIB_HANDLE[], :generate_csr_matrix)
    return ccall(func, Ptr{Cvoid}, (Cint, Cint, Cdouble, Cuint), m, n, sparsity, seed)
end

function c_allocate_result_csr(m::Int, max_nnz::Int)
    func = Libdl.dlsym(LIB_HANDLE[], :allocate_result_csr)
    return ccall(func, Ptr{Cvoid}, (Cint, Cint), m, max_nnz)
end

function c_reset_csr_result(mat::Ptr{Cvoid})
    func = Libdl.dlsym(LIB_HANDLE[], :reset_csr_result)
    ccall(func, Cvoid, (Ptr{Cvoid},), mat)
end

function c_hadamard_transpose(B::Ptr{Cvoid}, C::Ptr{Cvoid}, A::Ptr{Cvoid})
    func = Libdl.dlsym(LIB_HANDLE[], :hadamard_transpose)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), B, C, A)
end

function c_free_csr_matrix(mat::Ptr{Cvoid})
    func = Libdl.dlsym(LIB_HANDLE[], :free_csr_matrix)
    ccall(func, Cvoid, (Ptr{Cvoid},), mat)
end

function benchmark_comparison(m::Int, n::Int, sparsity::Float64)
    println("="^80)
    println("Benchmarking: A(i,j) = B(i,j) * C(j,i)")
    println("Matrix size: $m × $n")
    println("Sparsity: $(sparsity * 100)%")
    println("="^80)

    println("\n" * "="^80)
    println("FINCH BENCHMARK")
    println("="^80)

    println("\nGenerating matrices...")
    B_sparse = Utils.generate_sparse_matrix(m, n, sparsity, seed=42)
    C_sparse = Utils.generate_sparse_matrix(n, m, sparsity, seed=43)

    println("B nnz: $(SparseArrays.nnz(B_sparse))")
    println("C nnz: $(SparseArrays.nnz(C_sparse))")

    B_finch = Utils.to_finch_csr(B_sparse)
    C_finch = Utils.to_finch_csr(C_sparse)
    A_finch = Utils.allocate_finch_csr()
    # Warm up
    FinchKernels.hadamard_transpose!(A_finch, B_finch, C_finch)

    println("\nRunning Finch benchmark...")
    finch_result = BenchmarkTools.@benchmark FinchKernels.hadamard_transpose!($A_finch, $B_finch, $C_finch)

    println("\nFinch Results:")
    println("  Median time: ", BenchmarkTools.prettytime(BenchmarkTools.median(finch_result).time))
    println("  Mean time:   ", BenchmarkTools.prettytime(BenchmarkTools.mean(finch_result).time))
    println("  Min time:    ", BenchmarkTools.prettytime(minimum(finch_result).time))
    println("  Memory:      ", BenchmarkTools.prettymemory(BenchmarkTools.median(finch_result).memory))
    println("  Allocations: ", BenchmarkTools.median(finch_result).allocs)

    # === Unzipping Benchmark ===
    if LIB_HANDLE[] == C_NULL
        println("\n⚠ Unzipping library not found.")
        return (finch=finch_result, c=nothing, speedup=nothing)
    end

    println("\n" * "="^80)
    println("Unzipping BENCHMARK")
    println("="^80)

    println("\nGenerating matrices...")
    B_c = c_generate_csr_matrix(m, n, sparsity, UInt32(42))
    C_c = c_generate_csr_matrix(n, m, sparsity, UInt32(43))
    max_nnz = SparseArrays.nnz(B_sparse) + SparseArrays.nnz(C_sparse)
    A_c = c_allocate_result_csr(m, max_nnz)
    # Warm up
    c_hadamard_transpose(B_c, C_c, A_c)

    println("Running Unzipping benchmark...")

    c_result = BenchmarkTools.@benchmark begin
        c_reset_csr_result($A_c)
        c_hadamard_transpose($B_c, $C_c, $A_c)
    end

    println("\nUnzipping Results:")
    println("  Median time: ", BenchmarkTools.prettytime(BenchmarkTools.median(c_result).time))
    println("  Mean time:   ", BenchmarkTools.prettytime(BenchmarkTools.mean(c_result).time))
    println("  Min time:    ", BenchmarkTools.prettytime(minimum(c_result).time))
    println("  Memory:      ", BenchmarkTools.prettymemory(BenchmarkTools.median(c_result).memory))
    println("  Allocations: ", BenchmarkTools.median(c_result).allocs)

    c_free_csr_matrix(B_c)
    c_free_csr_matrix(C_c)
    c_free_csr_matrix(A_c)

    println("\n" * "="^80)
    println("COMPARISON")
    println("="^80)
    speedup = BenchmarkTools.median(finch_result).time / BenchmarkTools.median(c_result).time
    println("Speedup (Unzipping vs Finch): ", round(speedup, digits=2), "x")
    if speedup > 1.0
        println("✓ Unzipping is ", round(speedup, digits=2), "x faster than Finch")
    else
        println("✓ Finch is ", round(1 / speedup, digits=2), "x faster than Unzipping")
    end

    return (finch=finch_result, c=c_result, speedup=speedup)
end

function main()
    println("⚠ Compiling Unzipping library...")
    println("="^80)

    lib_ext = Sys.isapple() ? "dylib" : "so" # not sure about Windows
    run(`cc -shared -O3 -fPIC kernels.c utils.c -o libkernels.$lib_ext`)
    lib_path = joinpath(@__DIR__, "libkernels.$lib_ext")

    println("✓ Compiled successfully: $lib_path")
    println()

    if !isfile(lib_path)
        println("⚠ Unzipping library not found.")
        return
    end

    LIB_HANDLE[] = Libdl.dlopen(lib_path, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    println("✓ Loaded library: $lib_path")
    println()

    println("\n### Small Matrix Test ###")
    benchmark_comparison(100, 100, 0.1)

    println("\n\n### Medium Matrix Test ###")
    benchmark_comparison(1000, 1000, 0.05)

    println("\n\n### Large Matrix Test ###")
    benchmark_comparison(5000, 5000, 0.01)

    if LIB_HANDLE[] != C_NULL
        Libdl.dlclose(LIB_HANDLE[])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

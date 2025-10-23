#!/usr/bin/env julia

using BenchmarkTools
using CSV
using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

include("utils.jl")
include("kernels.jl")

const CONFIG = (
    sizes = [100, 200, 500, 1000, 2000, 3000, 5000],
    sparsities = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    fixed_size = 1000,
    fixed_sparsity = 0.05
)

# const CONFIG = (
#     sizes = [10, 20],
#     sparsities = [0.01, 0.02],
#     fixed_size = 10,
#     fixed_sparsity = 0.01
# )

# C Library Wrappers
const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function c_allocate_dense(size::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_dense)
    return ccall(func, Ptr{Cvoid}, (Csize_t,), size)
end

function c_free_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_reset_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_allocate_csr(ndim1::Csize_t, ndim2::Csize_t, nnz::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csr)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t), ndim1, ndim2, nnz)
end

function c_free_csr(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_csr)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_reset_csr(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_csr)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_generate_csr(ndim1::Csize_t, ndim2::Csize_t, sparsity::Cdouble, seed::Cuint)
    func = dlsym(LIB_HANDLE[], :generate_csr)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Cdouble, Cuint), ndim1, ndim2, sparsity, seed)
end

function c_allocate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, nnz2::Csize_t, nnz3::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csf)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), ndim1, ndim2, ndim3, nnz2, nnz3)
end

function c_free_csf(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_csf)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_reset_csf(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_csf)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function c_generate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, sparsity::Cdouble, seed::Cuint)
    func = dlsym(LIB_HANDLE[], :generate_csf)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Cdouble, Cuint), ndim1, ndim2, ndim3, sparsity, seed)
end

function c_hadamard_transpose(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :hadamard_transpose)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function c_matmul(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :matmul)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function c_matmul_hadamard(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, t3::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :matmul_hadamard)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, t3, res)
end

function c_hadamard_transpose_reduce(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :hadamard_transpose_reduce)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function c_permute_contract(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :permute_contract)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function save_results(results, kernel_name)
    rows = []
    
    # Process vary_size results
    sorted_sizes = sort(collect(keys(results["vary_size"])))
    for size in sorted_sizes
        group = results["vary_size"][size]
        finch_results = group["finch"]
        unzip_results = group["unzip"]
        
        push!(rows, (
            size = size,
            sparsity = CONFIG.fixed_sparsity,
            finch_min = minimum(finch_results).time,
            finch_med = median(finch_results).time,
            unzip_min = minimum(unzip_results).time,
            unzip_med = median(unzip_results).time,
        ))
    end
    
    # Process vary_sparsity results
    sorted_sparsities = sort(collect(keys(results["vary_sparsity"])))
    for sparsity in sorted_sparsities
        group = results["vary_sparsity"][sparsity]
        finch_results = group["finch"]
        unzip_results = group["unzip"]
        
        push!(rows, (
            size = CONFIG.fixed_size,
            sparsity = sparsity,
            finch_min = minimum(finch_results).time,
            finch_med = median(finch_results).time,
            unzip_min = minimum(unzip_results).time,
            unzip_med = median(unzip_results).time,
        ))
    end
    
    results_dir = "results"
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    CSV.write(joinpath(results_dir, "$(kernel_name).csv"), rows)
end

function run_benchmarks()
    # Compile and load library
    println("Compiling Unzipping kernels library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libkernels.$lib_ext"
    run(`cc -shared -O3 -fPIC kernels.c utils.c -o $lib_name`)
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    LIB_HANDLE[] = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()

    # --- Hadamard Transpose ---
    println("\n" * "="^80)
    println("BENCHMARKING: hadamard_transpose")
    println("="^80)
    suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = suite["vary_size"][size] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.hadamard_transpose($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = c_allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity)))
        group["unzip"] = @benchmarkable(c_hadamard_transpose($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.hadamard_transpose($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = c_allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity)))
        group["unzip"] = @benchmarkable(c_hadamard_transpose($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    results = run(suite, verbose=true)
    save_results(results, "hadamard_transpose")

    # --- Matrix Multiplication ---
    println("\n" * "="^80)
    println("BENCHMARKING: matmul")
    println("="^80)
    suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = suite["vary_size"][size] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.matmul($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = c_allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity * CONFIG.fixed_sparsity * size)))
        group["unzip"] = @benchmarkable(c_matmul($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.matmul($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = c_allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity * sparsity * CONFIG.fixed_size)))
        group["unzip"] = @benchmarkable(c_matmul($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    results = run(suite, verbose=true)
    save_results(results, "matmul")

    # --- Matrix Multiplication with Hadamard ---
    println("\n" * "="^80)
    println("BENCHMARKING: matmul_hadamard")
    println("="^80)
    suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = suite["vary_size"][size] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        D_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(44))
        group["finch"] = @benchmarkable FinchKernels.matmul_hadamard($B_finch, $C_finch, $D_finch)

        B_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        D_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(44))
        res_unzip = c_allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity * CONFIG.fixed_sparsity * size)))
        group["unzip"] = @benchmarkable(c_matmul_hadamard($B_unzip, $C_unzip, $D_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        D_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(44))
        group["finch"] = @benchmarkable FinchKernels.matmul_hadamard($B_finch, $C_finch, $D_finch)

        B_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        D_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(44))
        res_unzip = c_allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity * sparsity * CONFIG.fixed_size)))
        group["unzip"] = @benchmarkable(c_matmul_hadamard($B_unzip, $C_unzip, $D_unzip, $res_unzip), setup=(c_reset_csr($res_unzip)))
    end

    results = run(suite, verbose=true)
    save_results(results, "matmul_hadamard")

    # --- Hadamard Transpose Reduce ---
    println("\n" * "="^80)
    println("BENCHMARKING: hadamard_transpose_reduce")
    println("="^80)
    suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = suite["vary_size"][size] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.hadamard_transpose_reduce($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = c_allocate_dense(Csize_t(size))
        group["unzip"] = @benchmarkable(c_hadamard_transpose_reduce($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_dense($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = Utils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.hadamard_transpose_reduce($B_finch, $C_finch)

        B_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = c_generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = c_allocate_dense(Csize_t(CONFIG.fixed_size))
        group["unzip"] = @benchmarkable(c_hadamard_transpose_reduce($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_dense($res_unzip)))
    end

    results = run(suite, verbose=true)
    save_results(results, "hadamard_transpose_reduce")

    # --- Permute Contract ---
    println("\n" * "="^80)
    println("BENCHMARKING: permute_contract")
    println("="^80)
    benchmark_suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    benchmark_suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = benchmark_suite["vary_size"][size] = BenchmarkGroup()
        B_finch = Utils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = Utils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.permute_contract($B_finch, $C_finch)

        B_unzip = c_generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = c_generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = c_allocate_dense(Csize_t(size))
        group["unzip"] = @benchmarkable(c_permute_contract($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_dense($res_unzip)))
    end

    benchmark_suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = benchmark_suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = Utils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = Utils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch"] = @benchmarkable FinchKernels.permute_contract($B_finch, $C_finch)
        B_unzip = c_generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = c_generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = c_allocate_dense(Csize_t(CONFIG.fixed_size))
        group["unzip"] = @benchmarkable(c_permute_contract($B_unzip, $C_unzip, $res_unzip), setup=(c_reset_dense($res_unzip)))
    end

    println("DEBUG: Starting to run permute_contract benchmarks...")
    results = run(benchmark_suite, verbose=true)
    save_results(results, "permute_contract")

    dlclose(LIB_HANDLE[])
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end

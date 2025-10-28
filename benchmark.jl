#!/usr/bin/env julia

using BenchmarkTools
using CSV
using Finch

include("finch_utils.jl")
include("finch_kernels_jit.jl")
include("finch_kernels_aot.jl")
include("libunzip_utils.jl")
include("libunzip_kernels.jl")

const CONFIG = (
    sizes = [100, 200, 500, 1000],
    sparsities = [0.01, 0.02, 0.05, 0.10],
    fixed_size = 1000,
    fixed_sparsity = 0.05
)

# const CONFIG = (
#     sizes=[10, 20],
#     sparsities=[0.01, 0.02],
#     fixed_size=10,
#     fixed_sparsity=0.01
# )

function save_results(results, kernel_name)
    rows = []

    # Process vary_size results
    sorted_sizes = sort(collect(keys(results["vary_size"])))
    for size in sorted_sizes
        group = results["vary_size"][size]
        finch_jit_results = group["finch_jit"]
        finch_aot_results = group["finch_aot"]
        unzip_results = group["unzip"]

        push!(rows, (
            size=size,
            sparsity=CONFIG.fixed_sparsity,
            finch_jit_min=minimum(finch_jit_results).time,
            finch_aot_min=minimum(finch_aot_results).time,
            unzip_min=minimum(unzip_results).time,
            finch_jit_med=median(finch_jit_results).time,
            finch_aot_med=median(finch_aot_results).time,
            unzip_med=median(unzip_results).time,
        ))
    end

    # Process vary_sparsity results
    sorted_sparsities = sort(collect(keys(results["vary_sparsity"])))
    for sparsity in sorted_sparsities
        group = results["vary_sparsity"][sparsity]
        finch_jit_results = group["finch_jit"]
        finch_aot_results = group["finch_aot"]
        unzip_results = group["unzip"]

        push!(rows, (
            size=CONFIG.fixed_size,
            sparsity=sparsity,
            finch_jit_min=minimum(finch_jit_results).time,
            finch_aot_min=minimum(finch_aot_results).time,
            unzip_min=minimum(unzip_results).time,
            finch_jit_med=median(finch_jit_results).time,
            finch_aot_med=median(finch_aot_results).time,
            unzip_med=median(unzip_results).time,
        ))
    end

    results_dir = "results"
    if !isdir(results_dir)
        mkpath(results_dir)
    end
    CSV.write(joinpath(results_dir, "$(kernel_name).csv"), rows)
end

function run_benchmarks()
    UnzipUtils.setup()
    UnzipKernels.setup()

    # --- Hadamard Transpose ---
    println("\n" * "="^80)
    println("BENCHMARKING: hadamard_transpose")
    println("="^80)
    suite = BenchmarkGroup(["vary_size", "vary_sparsity"])

    suite["vary_size"] = BenchmarkGroup()
    for size in CONFIG.sizes
        group = suite["vary_size"][size] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity)))
        group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity)))
        group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
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
        B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity * CONFIG.fixed_sparsity * size)))
        group["unzip"] = @benchmarkable(UnzipKernels.matmul($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity * sparsity * CONFIG.fixed_size)))
        group["unzip"] = @benchmarkable(UnzipKernels.matmul($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
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
        B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        D_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(44))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        D_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(44))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * CONFIG.fixed_sparsity * CONFIG.fixed_sparsity * size)))
        group["unzip"] = @benchmarkable(UnzipKernels.matmul_hadamard($B_unzip, $C_unzip, $D_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        D_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(44))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        D_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(44))
        res_unzip = UnzipUtils.allocate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(floor(CONFIG.fixed_size * sparsity * sparsity * CONFIG.fixed_size)))
        group["unzip"] = @benchmarkable(UnzipKernels.matmul_hadamard($B_unzip, $C_unzip, $D_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
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
        B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_dense(Csize_t(size))
        group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose_reduce($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
    end

    suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))

        B_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csr(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_dense(Csize_t(CONFIG.fixed_size))
        group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose_reduce($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
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
        B_finch = FinchUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))

        B_unzip = UnzipUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(CONFIG.fixed_sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_dense(Csize_t(size))
        group["unzip"] = @benchmarkable(UnzipKernels.permute_contract($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
    end

    benchmark_suite["vary_sparsity"] = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        group = benchmark_suite["vary_sparsity"][sparsity] = BenchmarkGroup()
        B_finch = FinchUtils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_finch = FinchUtils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        group["finch_jit"] = @benchmarkable(FinchKernelsJIT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
        group["finch_aot"] = @benchmarkable(FinchKernelsAOT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
        B_unzip = UnzipUtils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(42))
        C_unzip = UnzipUtils.generate_csf(Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Csize_t(CONFIG.fixed_size), Cdouble(sparsity), Cuint(43))
        res_unzip = UnzipUtils.allocate_dense(Csize_t(CONFIG.fixed_size))
        group["unzip"] = @benchmarkable(UnzipKernels.permute_contract($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
    end

    println("DEBUG: Starting to run permute_contract benchmarks...")
    results = run(benchmark_suite, verbose=true)
    save_results(results, "permute_contract")

    UnzipKernels.teardown()
    UnzipUtils.teardown()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end

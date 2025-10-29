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
    sparsities = [0.01, 0.02, 0.05, 0.10],
    sizes = [100, 200, 500, 1000],
)

# DEBUG CONFIG
# const CONFIG = (
#     sparsities=[0.5, 0.8],
#     sizes=[10, 20],
# )

function save_results(results, kernel_name)
    rows = []
    for sparsity in CONFIG.sparsities
        for size in CONFIG.sizes
            group = results[sparsity][size]
            finch_jit_results = group["finch_jit"]
            finch_aot_results = group["finch_aot"]
            unzip_results = group["unzip"]

            push!(rows, (
                sparsity=sparsity,
                size=size,
                finch_jit_min=minimum(finch_jit_results).time / 1e6,
                finch_aot_min=minimum(finch_aot_results).time / 1e6,
                unzip_min=minimum(unzip_results).time / 1e6,
                finch_jit_med=median(finch_jit_results).time / 1e6,
                finch_aot_med=median(finch_aot_results).time / 1e6,
                unzip_med=median(unzip_results).time / 1e6,
            ))
        end
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

    k1_suite = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        sparsity_group = k1_suite[sparsity] = BenchmarkGroup()
        for size in CONFIG.sizes
            size_group = sparsity_group[size] = BenchmarkGroup()
            B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            size_group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
            size_group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

            B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * sparsity)))
            size_group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
        end
    end

    k1_results = BenchmarkTools.run(k1_suite, verbose=true)
    save_results(k1_results, "hadamard_transpose")

    # --- Matrix Multiplication ---
    println("\n" * "="^80)
    println("BENCHMARKING: matmul")
    println("="^80)

    k2_suite = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        sparsity_group = k2_suite[sparsity] = BenchmarkGroup()
        for size in CONFIG.sizes
            size_group = sparsity_group[size] = BenchmarkGroup()
            B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            size_group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
            size_group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul(A, $B_finch, $C_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

            B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * sparsity * sparsity * size) + 1))
            size_group["unzip"] = @benchmarkable(UnzipKernels.matmul($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
        end
    end

    k2_results = BenchmarkTools.run(k2_suite, verbose=true)
    save_results(k2_results, "matmul")

    # --- Matrix Multiplication with Hadamard ---
    println("\n" * "="^80)
    println("BENCHMARKING: matmul_hadamard")
    println("="^80)

    k3_suite = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        sparsity_group = k3_suite[sparsity] = BenchmarkGroup()
        for size in CONFIG.sizes
            size_group = sparsity_group[size] = BenchmarkGroup()
            B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            D_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(44))
            size_group["finch_jit"] = @benchmarkable(FinchKernelsJIT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))
            size_group["finch_aot"] = @benchmarkable(FinchKernelsAOT.matmul_hadamard(A, $B_finch, $C_finch, $D_finch), setup = (A = Tensor(SparseList(Dense(Element(0.0))))))

            B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            D_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(44))
            res_unzip = UnzipUtils.allocate_csr(Csize_t(size), Csize_t(size), Csize_t(floor(size * sparsity * sparsity * size) + 1))
            size_group["unzip"] = @benchmarkable(UnzipKernels.matmul_hadamard($B_unzip, $C_unzip, $D_unzip, $res_unzip), setup = (UnzipUtils.reset_csr($res_unzip)))
        end
    end

    k3_results = BenchmarkTools.run(k3_suite, verbose=true)
    save_results(k3_results, "matmul_hadamard")

    # --- Hadamard Transpose Reduce ---
    println("\n" * "="^80)
    println("BENCHMARKING: hadamard_transpose_reduce")
    println("="^80)

    k4_suite = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        sparsity_group = k4_suite[sparsity] = BenchmarkGroup()
        for size in CONFIG.sizes
            size_group = sparsity_group[size] = BenchmarkGroup()
            B_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_finch = FinchUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            size_group["finch_jit"] = @benchmarkable(FinchKernelsJIT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
            size_group["finch_aot"] = @benchmarkable(FinchKernelsAOT.hadamard_transpose_reduce(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))

            B_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_unzip = UnzipUtils.generate_csr(Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            res_unzip = UnzipUtils.allocate_dense(Csize_t(size))
            size_group["unzip"] = @benchmarkable(UnzipKernels.hadamard_transpose_reduce($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
        end
    end

    k4_results = BenchmarkTools.run(k4_suite, verbose=true)
    save_results(k4_results, "hadamard_transpose_reduce")

    # --- Permute Contract ---
    println("\n" * "="^80)
    println("BENCHMARKING: permute_contract")
    println("="^80)

    k5_suite = BenchmarkGroup()
    for sparsity in CONFIG.sparsities
        sparsity_group = k5_suite[sparsity] = BenchmarkGroup()
        for size in CONFIG.sizes
            size_group = sparsity_group[size] = BenchmarkGroup()
            B_finch = FinchUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_finch = FinchUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            size_group["finch_jit"] = @benchmarkable(FinchKernelsJIT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))
            size_group["finch_aot"] = @benchmarkable(FinchKernelsAOT.permute_contract(y, $B_finch, $C_finch), setup = (y = Tensor(Dense(Element(0.0)))))

            B_unzip = UnzipUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(42))
            C_unzip = UnzipUtils.generate_csf(Csize_t(size), Csize_t(size), Csize_t(size), Cdouble(sparsity), Cuint(43))
            res_unzip = UnzipUtils.allocate_dense(Csize_t(size))
            size_group["unzip"] = @benchmarkable(UnzipKernels.permute_contract($B_unzip, $C_unzip, $res_unzip), setup = (UnzipUtils.reset_dense($res_unzip)))
        end
    end

    k5_results = BenchmarkTools.run(k5_suite, verbose=true)
    save_results(k5_results, "permute_contract")

    UnzipKernels.teardown()
    UnzipUtils.teardown()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end

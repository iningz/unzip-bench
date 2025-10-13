# Utility functions for Finch tensor operations

module Utils

using Finch
using Random: rand, seed!
using SparseArrays: sparse, SparseMatrixCSC

"""
Generate a random sparse matrix with given size and sparsity.
"""
function generate_sparse_matrix(m::Int, n::Int, sparsity::Float64; seed=42)
    seed!(seed)
    nnz = Int(round(m * n * sparsity))
    I = rand(1:m, nnz)
    J = rand(1:n, nnz)
    V = rand(nnz)
    return sparse(I, J, V, m, n)
end

"""
Convert SparseMatrixCSC to Finch Tensor in CSR format.
CSR format: SparseList(Dense(Element(0.0)))
"""
function to_finch_csr(sp::SparseMatrixCSC)
    return Tensor(SparseList(Dense(Element(0.0))), sp)
end

"""
Allocate empty Finch tensor in CSR format.
"""
function allocate_finch_csr()
    return Tensor(SparseList(Dense(Element(0.0))))
end

"""
Compile the C/Unzipping library.
Returns the path to the compiled library.
"""
function compile_library()
    println("⚠ Compiling Unzipping library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libkernels.$lib_ext"
    lib_path = joinpath(@__DIR__, lib_name)

    run(`cc -shared -O3 -fPIC kernels.c utils.c -o $lib_name`)
    println("✓ Compiled successfully: $lib_path")

    if !isfile(lib_path)
        error("⚠ Library not found at: $lib_path")
    end

    return lib_path
end

end # module

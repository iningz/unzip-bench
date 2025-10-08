# Utility functions for Finch tensor operations

module Utils

import Finch
import SparseArrays: sparse, SparseMatrixCSC
import Random

"""
Generate a random sparse matrix with given size and sparsity.
"""
function generate_sparse_matrix(m::Int, n::Int, sparsity::Float64; seed=42)
    Random.seed!(seed)
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
    return Finch.Tensor(Finch.SparseList(Finch.Dense(Finch.Element(0.0))), sp)
end

"""
Allocate empty Finch tensor in CSR format.
"""
function allocate_finch_csr()
    return Finch.Tensor(Finch.SparseList(Finch.Dense(Finch.Element(0.0))))
end

end # module

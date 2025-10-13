module FinchKernels

using ..Utils: to_finch_csr
using Finch: @einsum
using SparseArrays: sparse

"""
Compute kernel: A(i, j) = B(i, j) * C(j, i)
"""
function hadamard_transpose!(B, C)
    @einsum A[i, j] = B[i, j] * C[j, i]
    return A
end

function test_hadamard_transpose()
    B_sparse = sparse([1, 1, 2], [1, 2, 2], [1.0, 2.0, 3.0], 2, 2)
    C_sparse = sparse([1, 2, 2], [1, 1, 2], [4.0, 5.0, 6.0], 2, 2)

    B = to_finch_csr(B_sparse)
    C = to_finch_csr(C_sparse)

    return hadamard_transpose!(B, C)
end

end # module

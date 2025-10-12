module FinchKernels

using Finch
import SparseArrays

"""
Compute kernel: A(i, j) = B(i, j) * C(j, i)
"""
function hadamard_transpose!(B, C)
    @einsum A[i, j] = B[i, j] * C[j, i]
    return A
end

function test_hadamard_transpose()
    B = Tensor(Dense(SparseList(Element(0.0))), [1.0 2.0; 0.0 3.0])
    C = Tensor(Dense(SparseList(Element(0.0))), [4.0 0.0; 5.0 6.0])
    return hadamard_transpose!(B, C)
end

end # module

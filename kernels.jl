# Kernel implementation using Finch
# A(i,j) = B(i,j) * C(j,i)

module FinchKernels

import Finch

"""
Compute kernel: A(i, j) = B(i, j) * C(j, i)
"""
function hadamard_transpose!(A, B, C)
    Finch.@einsum A[i, j] = B[i, j] * C[j, i]
    return A
end

end # module

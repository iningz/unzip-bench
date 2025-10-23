module FinchKernels
export hadamard_transpose, matmul, matmul_hadamard, hadamard_transpose_reduce, permute_contract

using Finch

# A(i, j) = B(i, j) * C(j, i)
function hadamard_transpose(B, C)
    @einsum A[i, j] = B[i, j] * C[j, i]
    return A
end

# A(i, j) = B(i, k) * C(k, j)
function matmul(B, C)
    @einsum A[i, j] += B[i, k] * C[k, j]
    return A
end

# A(i, j) = B(i, k) * C(k, j) * D(k, j)
function matmul_hadamard(B, C, D)
    @einsum A[i, j] += B[i, k] * C[k, j] * D[k, j]
    return A
end

# y(i) = B(i, j) * C(j, i)
function hadamard_transpose_reduce(B, C)
    @einsum y[i] += B[i, j] * C[j, i]
    return y
end

# y(i) = B(i, j, k) * C(i, k, j)
function permute_contract(B, C)
    @einsum y[i] += B[i, j, k] * C[i, k, j]
    return y
end

end # module

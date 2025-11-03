module FinchKernelsAOT

using Finch

export hadamard_transpose, matmul, matmul_hadamard, hadamard_transpose_reduce, permute_contract

y = Tensor(Dense(Element(0.0)))
A = Tensor(SparseList(Dense(Element(0.0))))
B = Tensor(SparseList(Dense(Element(0.0))))
C = Tensor(SparseList(Dense(Element(0.0))))
D = Tensor(SparseList(Dense(Element(0.0))))
B_3d = Tensor(Dense(SparseList(SparseList(Element(0.0)))))
C_3d = Tensor(Dense(SparseList(SparseList(Element(0.0)))))

# A(i, j) = B(i, j) * C(j, i)
hadamard_transpose_def = @finch_kernel mode = :fast function hadamard_transpose(A, B, C)
    A .= 0
    for j = _, i = _
        A[i, j] = B[i, j] * C[j, i]
    end
    return A
end
eval(hadamard_transpose_def)

# A(i, j) = B(i, k) * C(k, j)
matmul_def = @finch_kernel mode = :fast function matmul(A, B, C)
    A .= 0
    for j = _, k = _, i = _
        A[i, j] += B[i, k] * C[k, j]
    end
    return A
end
eval(matmul_def)

# A(i, j) = B(i, k) * C(k, j) * D(k, j)
matmul_hadamard_def = @finch_kernel mode = :fast function matmul_hadamard(A, B, C, D)
    A .= 0
    for j = _, k = _, i = _
        A[i, j] += B[i, k] * C[k, j] * D[k, j]
    end
    return A
end
eval(matmul_hadamard_def)

# y(i) = B(i, j) * C(j, i)
hadamard_transpose_reduce_def = @finch_kernel mode = :fast function hadamard_transpose_reduce(y, B, C)
    y .= 0
    for j = _, i = _
        y[i] += B[i, j] * C[j, i]
    end
    return y
end
eval(hadamard_transpose_reduce_def)

# y(i) = B(i, j, k) * C(i, k, j)
permute_contract_def = @finch_kernel mode = :fast function permute_contract(y, B_3d, C_3d)
    y .= 0
    for k = _, j = _, i = _
        y[i] += B_3d[i, j, k] * C_3d[i, k, j]
    end
    return y
end
eval(permute_contract_def)

end # module

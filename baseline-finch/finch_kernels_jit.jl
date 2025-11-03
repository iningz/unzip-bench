module FinchKernelsJIT
export hadamard_transpose, matmul, matmul_hadamard, hadamard_transpose_reduce, permute_contract

using Finch

# A(i, j) = B(i, j) * C(j, i)
function hadamard_transpose(A, B, C)
    @finch mode = :fast begin
        A .= 0
        for j = _, i = _
            A[i, j] = B[i, j] * C[j, i]
        end
    end
    return A
end

# A(i, j) = B(i, k) * C(k, j)
function matmul(A, B, C)
    @finch mode = :fast begin
        A .= 0
        for j = _, k = _, i = _
            A[i, j] += B[i, k] * C[k, j]
        end
    end
    return A
end

# A(i, j) = B(i, k) * C(k, j) * D(k, j)
function matmul_hadamard(A, B, C, D)
    @finch mode = :fast begin
        A .= 0
        for j = _, k = _, i = _
            A[i, j] += B[i, k] * C[k, j] * D[k, j]
        end
    end
    return A
end

# y(i) = B(i, j) * C(j, i)
function hadamard_transpose_reduce(y, B, C)
    @finch mode = :fast begin
        y .= 0
        for j = _, i = _
            y[i] += B[i, j] * C[j, i]
        end
    end
    return y
end

# y(i) = B(i, j, k) * C(i, k, j)
function permute_contract(y, B, C)
    @finch mode = :fast begin
        y .= 0
        for k = _, j = _, i = _
            y[i] += B[i, j, k] * C[i, k, j]
        end
    end
    return y
end

end # module

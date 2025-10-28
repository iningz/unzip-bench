module FinchKernelsTest
export test_hadamard_transpose, test_matmul, test_matmul_hadamard, test_hadamard_transpose_reduce, test_permute_contract

using Finch

function test_hadamard_transpose(Kernels)
    # B = [1 2; 0 3]
    B_rows = [1, 1, 2]
    B_cols = [1, 2, 2]
    B_vals = [1.0, 2.0, 3.0]
    B_coo = fsparse(B_rows, B_cols, B_vals, (2, 2))
    B = Tensor(SparseList(Dense(Element(0.0))), B_coo)

    # C = [4 0; 5 6]
    C_rows = [1, 2, 2]
    C_cols = [1, 1, 2]
    C_vals = [4.0, 5.0, 6.0]
    C_coo = fsparse(C_rows, C_cols, C_vals, (2, 2))
    C = Tensor(SparseList(Dense(Element(0.0))), C_coo)

    # A(i,j) = B(i,j) * C(j,i)
    # Expected: A = [4 10; 0 18]
    A = Tensor(SparseList(Dense(Element(0.0))))
    display(Kernels.hadamard_transpose(A, B, C))
end

function test_matmul(Kernels)
    # B = [1 1; 0 0]
    B_rows = [1, 1]
    B_cols = [1, 2]
    B_vals = [1.0, 1.0]
    B_coo = fsparse(B_rows, B_cols, B_vals, (2, 2))
    B = Tensor(SparseList(Dense(Element(0.0))), B_coo)

    # C = [1 0; 1 0]
    C_rows = [1, 2]
    C_cols = [1, 1]
    C_vals = [1.0, 1.0]
    C_coo = fsparse(C_rows, C_cols, C_vals, (2, 2))
    C = Tensor(SparseList(Dense(Element(0.0))), C_coo)

    # A(i,j) = B(i,k) * C(k,j)
    # Expected: A = [2 0; 0 0]
    A = Tensor(SparseList(Dense(Element(0.0))))
    display(Kernels.matmul(A, B, C))
end

function test_matmul_hadamard(Kernels)
    # B = [1 2; 0 3]
    B_rows = [1, 1, 2]
    B_cols = [1, 2, 2]
    B_vals = [1.0, 2.0, 3.0]
    B_coo = fsparse(B_rows, B_cols, B_vals, (2, 2))
    B = Tensor(SparseList(Dense(Element(0.0))), B_coo)

    # C = [1 0; 0 1]
    C_rows = [1, 2]
    C_cols = [1, 2]
    C_vals = [1.0, 1.0]
    C_coo = fsparse(C_rows, C_cols, C_vals, (2, 2))
    C = Tensor(SparseList(Dense(Element(0.0))), C_coo)

    # D = [2 0; 0 2]
    D_rows = [1, 2]
    D_cols = [1, 2]
    D_vals = [2.0, 2.0]
    D_coo = fsparse(D_rows, D_cols, D_vals, (2, 2))
    D = Tensor(SparseList(Dense(Element(0.0))), D_coo)

    # A(i,j) = B(i,k) * C(k,j) * D(k,j)
    # Expected: A = [2 0; 0 6]
    A = Tensor(SparseList(Dense(Element(0.0))))
    display(Kernels.matmul_hadamard(A, B, C, D))
end

function test_hadamard_transpose_reduce(Kernels)
    # B = [1 2; 0 3]
    B_rows = [1, 1, 2]
    B_cols = [1, 2, 2]
    B_vals = [1.0, 2.0, 3.0]
    B_coo = fsparse(B_rows, B_cols, B_vals, (2, 2))
    B = Tensor(SparseList(Dense(Element(0.0))), B_coo)

    # C = [4 0; 5 6]
    C_rows = [1, 2, 2]
    C_cols = [1, 1, 2]
    C_vals = [4.0, 5.0, 6.0]
    C_coo = fsparse(C_rows, C_cols, C_vals, (2, 2))
    C = Tensor(SparseList(Dense(Element(0.0))), C_coo)

    # y(i) = sum_j B(i,j) * C(j,i)
    # Expected: y = [14, 18]
    y = Tensor(Dense(Element(0.0)))
    display(Kernels.hadamard_transpose_reduce(y, B, C))
end

function test_permute_contract(Kernels)
    # B has non-zeros at: B[1,1,1]=1, B[1,1,2]=2, B[2,2,2]=3
    B_i = [1, 1, 2]
    B_j = [1, 1, 2]
    B_k = [1, 2, 2]
    B_vals = [1.0, 2.0, 3.0]
    B_coo = fsparse(B_i, B_j, B_k, B_vals, (2, 2, 2))
    B = Tensor(Dense(SparseList(SparseList(Element(0.0)))), B_coo)

    # C has non-zeros at: C[1,1,1]=4, C[2,1,2]=5, C[2,2,2]=6
    C_i = [1, 2, 2]
    C_k = [1, 1, 2]
    C_j = [1, 2, 2]
    C_vals = [4.0, 5.0, 6.0]
    C_coo = fsparse(C_i, C_k, C_j, C_vals, (2, 2, 2))
    C = Tensor(Dense(SparseList(SparseList(Element(0.0)))), C_coo)

    # y(i) = sum_jk B(i,j,k) * C(i,k,j)
    # Expected: y = [4, 18]
    y = Tensor(Dense(Element(0.0)))
    display(Kernels.permute_contract(y, B, C))
end

end # module

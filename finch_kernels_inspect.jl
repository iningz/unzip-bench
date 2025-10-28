using Finch

function inspect_code()
    y = Tensor(Dense(Element(0.0)))
    A = Tensor(SparseList(Dense(Element(0.0))))
    B = Tensor(SparseList(Dense(Element(0.0))))
    C = Tensor(SparseList(Dense(Element(0.0))))
    D = Tensor(SparseList(Dense(Element(0.0))))
    B_3d = Tensor(Dense(SparseList(SparseList(Element(0.0)))))
    C_3d = Tensor(Dense(SparseList(SparseList(Element(0.0)))))

    println("\n--- hadamard_transpose ---")
    println("A(i, j) = B(i, j) * C(j, i)")
    display(@finch_code mode = :fast (A .= 0;
    for j = _, i = _
        A[i, j] = B[i, j] * C[j, i]
    end))


    println("\n--- matmul ---")
    println("A(i, j) = B(i, k) * C(k, j)")
    display(@finch_code mode = :fast (A .= 0;
    for j = _, k = _, i = _
        A[i, j] += B[i, k] * C[k, j]
    end))


    println("\n--- matmul_hadamard ---")
    println("A(i, j) = B(i, k) * C(k, j) * D(k, j)")
    display(@finch_code mode = :fast (A .= 0;
    for j = _, k = _, i = _
        A[i, j] += B[i, k] * C[k, j] * D[k, j]
    end))


    println("\n--- hadamard_transpose_reduce ---")
    println("y(i) = B(i, j) * C(j, i)")
    display(@finch_code mode = :fast (y .= 0;
    for j = _, i = _
        y[i] += B[i, j] * C[j, i]
    end))


    println("\n--- permute_contract ---")
    println("y(i) = B(i, j, k) * C(i, k, j)")
    display(@finch_code mode = :fast (y .= 0;
    for k = _, j = _, i = _
        y[i] += B_3d[i, j, k] * C_3d[i, k, j]
    end))
end

if abspath(PROGRAM_FILE) == @__FILE__
    inspect_code()
end
# Test correctness of Finch vs Unzipping implementations

include("finch_kernels_jit.jl")
include("finch_kernels_aot.jl")
include("libunzip_kernels_test.jl")
include("finch_kernels_test.jl")


function run_tests()
    UnzipKernelsTest.setup()

    println("="^80)
    println("TEST 1: Hadamard Transpose")
    println("B = [1 2; 0 3]")
    println("C = [4 0; 5 6]")
    println("A(i,j) = B(i,j) * C(j,i)")
    println("Expected: A = [4 10; 0 18]")

    println("\n------ Finch JIT Result ------")
    FinchKernelsTest.test_hadamard_transpose(FinchKernelsJIT)
    println()

    println("\n------ Finch AOT Result ------")
    FinchKernelsTest.test_hadamard_transpose(FinchKernelsAOT)
    println()

    println("\n------ Unzipping Result ------")
    UnzipKernelsTest.test_hadamard_transpose()
    println("="^80)
    println()

    println("="^80)
    println("TEST 2: Matrix Multiplication")
    println("B = [1 1; 0 0]")
    println("C = [1 0; 1 0]")
    println("A(i,j) = B(i,k) * C(k,j)")
    println("Expected: A = [2 0; 0 0]")

    println("\n------ Finch JIT Result ------")
    FinchKernelsTest.test_matmul(FinchKernelsJIT)
    println()

    println("\n------ Finch AOT Result ------")
    FinchKernelsTest.test_matmul(FinchKernelsAOT)
    println()

    println("\n------ Unzipping Result ------")
    UnzipKernelsTest.test_matmul()
    println("="^80)
    println()

    println("="^80)
    println("TEST 3: Matrix Multiplication with Hadamard")
    println("B = [1 2; 0 3]")
    println("C = [1 0; 0 1]")
    println("D = [2 0; 0 2]")
    println("A(i,j) = B(i,k) * C(k,j) * D(k,j)")
    println("Expected: A = [2 0; 0 6]")

    println("\n------ Finch JIT Result ------")
    FinchKernelsTest.test_matmul_hadamard(FinchKernelsJIT)
    println()

    println("\n------ Finch AOT Result ------")
    FinchKernelsTest.test_matmul_hadamard(FinchKernelsAOT)
    println()

    println("\n------ Unzipping Result ------")
    UnzipKernelsTest.test_matmul_hadamard()
    println("="^80)
    println()

    println("="^80)
    println("TEST 4: Hadamard Transpose Reduce")
    println("B = [1 2; 0 3]")
    println("C = [4 0; 5 6]")
    println("y(i) = B(i,j) * C(j,i)")
    println("Expected: y = [14, 18]")

    println("\n------ Finch JIT Result ------")
    FinchKernelsTest.test_hadamard_transpose_reduce(FinchKernelsJIT)
    println()

    println("\n------ Finch AOT Result ------")
    FinchKernelsTest.test_hadamard_transpose_reduce(FinchKernelsAOT)
    println()

    println("\n------ Unzipping Result ------")
    UnzipKernelsTest.test_hadamard_transpose_reduce()
    println("="^80)
    println()

    println("="^80)
    println("TEST 5: 3D Tensor Permute Contract")
    println("y(i) = sum_jk B(i,j,k) * C(i,k,j)")
    println("B has non-zeros at: B[0,0,0]=1, B[0,0,1]=2, B[1,1,1]=3")
    println("C has non-zeros at: C[0,0,0]=4, C[1,0,1]=5, C[1,1,1]=6")
    println("Expected: y = [4, 18]")

    println("\n------ Finch JIT Result ------")
    FinchKernelsTest.test_permute_contract(FinchKernelsJIT)
    println()

    println("\n------ Finch AOT Result ------")
    FinchKernelsTest.test_permute_contract(FinchKernelsAOT)
    println()

    println("\n------ Unzipping Result ------")
    UnzipKernelsTest.test_permute_contract()
    println("="^80)
    println()

    UnzipKernelsTest.teardown()
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

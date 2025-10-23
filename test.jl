# Test correctness of Finch vs Unzipping implementations

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

include("kernels_test.jl")
using .FinchKernelsTest

function run_tests()
    # Compile and load library
    println("Compiling Unzipping test library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libkernel_tests.$lib_ext"
    run(`cc -shared -O3 -fPIC kernels_test.c kernels.c utils.c -o $lib_name`)
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    lib_handle = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()

    println("="^80)
    println("TEST 1: Hadamard Transpose")
    println("B = [1 2; 0 3]")
    println("C = [4 0; 5 6]")
    println("A(i,j) = B(i,j) * C(j,i)")
    println("Expected: A = [4 10; 0 18]")

    println("\n------ Finch Result ------")
    test_hadamard_transpose()
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_hadamard_transpose)
    ccall(unzipping_test_func, Cvoid, ())
    println("="^80)
    println()

    println("="^80)
    println("TEST 2: Matrix Multiplication")
    println("B = [1 1; 0 0]")
    println("C = [1 0; 1 0]")
    println("A(i,j) = B(i,k) * C(k,j)")
    println("Expected: A = [2 0; 0 0]")

    println("\n------ Finch Result ------")
    test_matmul()
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_matmul)
    ccall(unzipping_test_func, Cvoid, ())
    println("="^80)
    println()

    println("="^80)
    println("TEST 3: Matrix Multiplication with Hadamard")
    println("B = [1 2; 0 3]")
    println("C = [1 0; 0 1]")
    println("D = [2 0; 0 2]")
    println("A(i,j) = B(i,k) * C(k,j) * D(k,j)")
    println("Expected: A = [2 0; 0 6]")

    println("\n------ Finch Result ------")
    test_matmul_hadamard()
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_matmul_hadamard)
    ccall(unzipping_test_func, Cvoid, ())
    println("="^80)
    println()

    println("="^80)
    println("TEST 4: Hadamard Transpose Reduce")
    println("B = [1 2; 0 3]")
    println("C = [4 0; 5 6]")
    println("y(i) = B(i,j) * C(j,i)")
    println("Expected: y = [14, 18]")

    println("\n------ Finch Result ------")
    test_hadamard_transpose_reduce()
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_hadamard_transpose_reduce)
    ccall(unzipping_test_func, Cvoid, ())
    println("="^80)
    println()

    println("="^80)
    println("TEST 5: 3D Tensor Permute Contract")
    println("y(i) = sum_jk B(i,j,k) * C(i,k,j)")
    println("B has non-zeros at: B[0,0,0]=1, B[0,0,1]=2, B[1,1,1]=3")
    println("C has non-zeros at: C[0,0,0]=4, C[1,0,1]=5, C[1,1,1]=6")
    println("Expected: y = [4, 18]")

    println("\n------ Finch Result ------")
    test_permute_contract()
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_permute_contract)
    ccall(unzipping_test_func, Cvoid, ())
    println("="^80)
    println()

    dlclose(lib_handle)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

# Test correctness of Finch vs Unzipping implementations

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

include("utils.jl")
include("kernels.jl")

function run_tests()
    # Compile and load library
    lib_path = Utils.compile_library()
    lib_handle = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("✓ Loaded library: $lib_path")
    println()

    println("="^80)
    println("CORRECTNESS TEST")
    println("B = [1 2; 0 3]")
    println("C = [4 0; 5 6]")
    println("A(i,j) = B(i,j) * C(j,i)")

    println("\n------ Finch Result ------")
    finch_result = FinchKernels.test_hadamard_transpose()
    display(finch_result)
    println()

    println("\n------ Unzipping Result ------")
    unzipping_test_func = dlsym(lib_handle, :test_hadamard_transpose)
    ccall(unzipping_test_func, Cvoid, ())

    println("="^80)

    dlclose(lib_handle)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end

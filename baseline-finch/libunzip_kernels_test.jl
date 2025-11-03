module UnzipKernelsTest

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

export setup, teardown, test_hadamard_transpose, test_matmul, test_matmul_hadamard, test_hadamard_transpose_reduce, test_permute_contract

const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function setup()
    println("Compiling Unzipping kernels test library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libunzip_kernels_test.$(lib_ext)"
    run(`cc -shared -O3 -fPIC unzip_kernels_test.c unzip_kernels.c -o $lib_name`)
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    LIB_HANDLE[] = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()
end

function teardown()
    dlclose(LIB_HANDLE[])
end

function test_hadamard_transpose()
    func = dlsym(LIB_HANDLE[], :test_hadamard_transpose)
    ccall(func, Cvoid, ())
end

function test_matmul()
    func = dlsym(LIB_HANDLE[], :test_matmul)
    ccall(func, Cvoid, ())
end

function test_matmul_hadamard()
    func = dlsym(LIB_HANDLE[], :test_matmul_hadamard)
    ccall(func, Cvoid, ())
end

function test_hadamard_transpose_reduce()
    func = dlsym(LIB_HANDLE[], :test_hadamard_transpose_reduce)
    ccall(func, Cvoid, ())
end

function test_permute_contract()
    func = dlsym(LIB_HANDLE[], :test_permute_contract)
    ccall(func, Cvoid, ())
end

end # module

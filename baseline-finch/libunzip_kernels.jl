module UnzipKernels

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

export setup, teardown, hadamard_transpose, matmul, matmul_hadamard, hadamard_transpose_reduce, permute_contract

const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function setup()
    println("Compiling Unzipping kernels library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libunzip_kernels.$(lib_ext)"
    run(`cc -shared -O3 -fPIC unzip_kernels.c -o $lib_name`)
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    LIB_HANDLE[] = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()
end

function teardown()
    dlclose(LIB_HANDLE[])
end

function hadamard_transpose(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :hadamard_transpose)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function matmul(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :matmul)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function matmul_hadamard(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, t3::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :matmul_hadamard)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, t3, res)
end

function hadamard_transpose_reduce(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :hadamard_transpose_reduce)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

function permute_contract(t1::Ptr{Cvoid}, t2::Ptr{Cvoid}, res::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :permute_contract)
    ccall(func, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), t1, t2, res)
end

end # module

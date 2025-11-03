module UnzipKernels

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

export setup, teardown, hadamard_transpose, matmul, matmul_hadamard, hadamard_transpose_reduce, permute_contract, allocate_dense, free_dense, reset_dense, allocate_csr, free_csr, reset_csr, generate_csr, allocate_csf, free_csf, reset_csf, generate_csf

const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function setup(lib_basename)
    println("Compiling Unzipping kernels library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "$(lib_basename).$(lib_ext)"
    if lib_basename == "libunzip"
        run(`cc -shared -O3 -fPIC unzip_kernels.c unzip_utils.c -o $lib_name`)
    elseif lib_basename == "libunzip_test"
        run(`cc -shared -O3 -fPIC unzip_kernels_test.c unzip_kernels.c unzip_utils.c -o $lib_name`)
    else
        error("Unknown library to compile: $lib_basename")
    end
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    LIB_HANDLE[] = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()
end

function teardown()
    dlclose(LIB_HANDLE[])
end

function allocate_dense(size::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_dense)
    return ccall(func, Ptr{Cvoid}, (Csize_t,), size)
end

function free_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function reset_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function allocate_csr(ndim1::Csize_t, ndim2::Csize_t, nnz::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csr)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t), ndim1, ndim2, nnz)
end

function free_csr(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_csr)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function reset_csr(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_csr)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function generate_csr(ndim1::Csize_t, ndim2::Csize_t, sparsity::Cdouble, seed::Cuint)
    func = dlsym(LIB_HANDLE[], :generate_csr)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Cdouble, Cuint), ndim1, ndim2, sparsity, seed)
end

function allocate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, nnz2::Csize_t, nnz3::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csf)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), ndim1, ndim2, ndim3, nnz2, nnz3)
end

function free_csf(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_csf)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function reset_csf(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_csf)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function generate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, sparsity::Cdouble, seed::Cuint)
    func = dlsym(LIB_HANDLE[], :generate_csf)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Cdouble, Cuint), ndim1, ndim2, ndim3, sparsity, seed)
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

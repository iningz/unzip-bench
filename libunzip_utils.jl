module UnzipUtils

using Libdl: dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_GLOBAL

export setup, teardown, allocate_dense, free_dense, reset_dense, allocate_csr, free_csr, reset_csr, generate_csr, allocate_csf, free_csf, reset_csf, generate_csf

const LIB_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

function setup()
    println("Compiling Unzipping utils library...")
    lib_ext = Sys.isapple() ? "dylib" : "so"
    lib_name = "libunzip_utils.$(lib_ext)"
    run(`cc -shared -O3 -fPIC unzip_utils.c -o $lib_name`)
    println("Compiled library: $lib_name")
    lib_path = joinpath(@__DIR__, lib_name)
    LIB_HANDLE[] = dlopen(lib_path, RTLD_LAZY | RTLD_GLOBAL)
    println("Loaded library: $lib_path")
    println()
end

function teardown()
    dlclose(LIB_HANDLE[])
end

function allocate_dense(n::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_dense)
    return ccall(func, Ptr{Cvoid}, (Csize_t,), n)
end

function free_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :free_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function reset_dense(tensor::Ptr{Cvoid})
    func = dlsym(LIB_HANDLE[], :reset_dense)
    ccall(func, Cvoid, (Ptr{Cvoid},), tensor)
end

function allocate_csr(ndim1::Csize_t, ndim2::Csize_t, dim2_nnz::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csr)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t), ndim1, ndim2, dim2_nnz)
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

function allocate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, dim2_nnz::Csize_t, dim3_nnz::Csize_t)
    func = dlsym(LIB_HANDLE[], :allocate_csf)
    return ccall(func, Ptr{Cvoid}, (Csize_t, Csize_t, Csize_t, Csize_t, Csize_t), ndim1, ndim2, ndim3, dim2_nnz, dim3_nnz)
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

end # module

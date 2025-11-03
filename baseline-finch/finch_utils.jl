# Utility functions for Finch tensor operations

module FinchUtils

using Finch
using Random

# Generate a random sparse matrix in CSR format (SparseList(Dense(Element(0.0)))).
function generate_csr(ndim1::Csize_t, ndim2::Csize_t, sparsity::Cdouble, seed::Cuint)
    Random.seed!(seed)

    # Calculate number of non-zeros per row
    dim2_nnz = Csize_t(floor(ndim2 * sparsity))
    if dim2_nnz < 1
        dim2_nnz = 1
    end
    if dim2_nnz > ndim2
        dim2_nnz = ndim2
    end

    # Total non-zeros: each row has exactly dim2_nnz non-zeros
    total_nnz = ndim1 * dim2_nnz

    # Generate random coordinate arrays and values
    dim1_crds = Vector{Int}(undef, total_nnz)
    dim2_crds = Vector{Int}(undef, total_nnz)
    vals = Vector{Cdouble}(undef, total_nnz)

    for dim1_idx in 1:ndim1
        # Generate exactly dim2_nnz non-zeros per row
        for n in 1:dim2_nnz
            idx = (dim1_idx - 1) * dim2_nnz + n
            dim1_crds[idx] = dim1_idx
            dim2_crds[idx] = rand(1:ndim2)
            vals[idx] = rand(Cdouble)
        end
    end

    # Create COO tensor using fsparse, then convert to CSR format
    coo_tensor = fsparse(dim1_crds, dim2_crds, vals, (Int(ndim1), Int(ndim2)))
    csr_tensor = Tensor(SparseList(Dense(Element(0.0))), coo_tensor)

    return csr_tensor
end

# Generate a random sparse 3D tensor in CSF format ((Dense(SparseList(SparseList((Element(0.0))))).
function generate_csf(ndim1::Csize_t, ndim2::Csize_t, ndim3::Csize_t, sparsity::Cdouble, seed::Cuint)
    Random.seed!(seed)

    # Calculate number of non-zeros per dimension
    dim2_nnz = Csize_t(floor(ndim2 * sparsity))
    if dim2_nnz < 1
        dim2_nnz = 1
    end
    if dim2_nnz > ndim2
        dim2_nnz = ndim2
    end

    dim3_nnz = Csize_t(floor(ndim3 * sparsity))
    if dim3_nnz < 1
        dim3_nnz = 1
    end
    if dim3_nnz > ndim3
        dim3_nnz = ndim3
    end

    # Total non-zeros: each dim1 slice has dim2_nnz fibers, each fiber has dim3_nnz elements
    dim2_total = ndim1 * dim2_nnz  # total fibers
    dim3_total = dim2_total * dim3_nnz  # total elements

    # Generate random coordinate arrays and values
    dim1_crds = Vector{Int}(undef, dim3_total)
    dim2_crds = Vector{Int}(undef, dim3_total)
    dim3_crds = Vector{Int}(undef, dim3_total)
    vals = Vector{Cdouble}(undef, dim3_total)

    for dim1_idx in 1:ndim1
        # Generate exactly dim2_nnz fibers per slice
        for dim2_n in 1:dim2_nnz
            dim2_idx = (dim1_idx - 1) * dim2_nnz + dim2_n
            dim2_crd = rand(1:ndim2)

            # Generate exactly dim3_nnz elements per fiber
            for dim3_n in 1:dim3_nnz
                idx = (dim2_idx - 1) * dim3_nnz + dim3_n
                dim1_crds[idx] = dim1_idx
                dim2_crds[idx] = dim2_crd
                dim3_crds[idx] = rand(1:ndim3)
                vals[idx] = rand(Cdouble)
            end
        end
    end

    # Create COO tensor using fsparse, then convert to CSF format
    coo_tensor = fsparse(dim1_crds, dim2_crds, dim3_crds, vals, (Int(ndim1), Int(ndim2), Int(ndim3)))
    csf_tensor = Tensor(Dense(SparseList(SparseList(Element(0.0)))), coo_tensor)

    return csf_tensor
end

end # module

#include "tensor_formats.h"
#include <stdlib.h>
#include <string.h>

static inline double rand_double() { return (double)rand() / RAND_MAX; }

static inline size_t rand_size_t() {
  size_t result = 0;
  for (size_t i = 0; i < sizeof(size_t); ++i)
    result = (result << 8) | (rand() & 0xFF);
  return result;
}

static inline size_t rand_uniform(size_t n) { return rand_size_t() % n; }

// ============================================================================
// Dense tensor utilities
// ============================================================================

struct dense *_allocate_dense(size_t n) {
  struct dense *tensor = malloc(sizeof(struct dense));
  tensor->lvl1_size = n;
  tensor->vals = calloc(n, sizeof(double));
  return tensor;
}

void _free_dense(struct dense *tensor) {
  if (tensor) {
    free(tensor->vals);
    free(tensor);
  }
}

void _reset_dense(struct dense *tensor) { memset(tensor->vals, 0, tensor->lvl1_size * sizeof(double)); }

struct dense *_generate_dense(size_t n, unsigned int seed) {
  srand(seed);
  struct dense *tensor = _allocate_dense(n);
  for (size_t i = 0; i < n; ++i) {
    tensor->vals[i] = rand_double();
  }
  return tensor;
}

// ============================================================================
// CSR tensor utilities
// ============================================================================

struct csr *_allocate_csr(size_t ndim1, size_t dim2_nnz) {
  struct csr *tensor = malloc(sizeof(struct csr));
  tensor->lvl1_size = ndim1;
  tensor->lvl2_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  tensor->lvl2_nnz = ndim1 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl2_nnz, sizeof(double));
  return tensor;
}

void _free_csr(struct csr *tensor) {
  if (tensor) {
    free(tensor->lvl2_pos);
    free(tensor->lvl2_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void _reset_csr(struct csr *tensor) {
  tensor->lvl2_nnz = 0;
  memset(tensor->lvl2_pos, 0, (tensor->lvl1_size + 1) * sizeof(size_t));
}

struct csr *_generate_csr(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed) {
  srand(seed);

  struct csr *tensor = malloc(sizeof(struct csr));
  tensor->lvl1_size = ndim1;
  tensor->lvl2_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  size_t dim2_nnz = (size_t)(ndim2 * sparsity);
  if (dim2_nnz < 1)
    dim2_nnz = 1;
  if (dim2_nnz > ndim2)
    dim2_nnz = ndim2;

  tensor->lvl2_nnz = ndim1 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl2_nnz, sizeof(double));

  for (size_t lvl1_idx = 0; lvl1_idx < ndim1; ++lvl1_idx) {
    // Generate exactly dim2_nnz non-zeros per row
    for (size_t dim2_n = 0; dim2_n < dim2_nnz; ++dim2_n) {
      size_t lvl2_idx = lvl1_idx * dim2_nnz + dim2_n;
      tensor->lvl2_crd[lvl2_idx] = rand_uniform(ndim2);
      tensor->vals[lvl2_idx] = rand_double();
    }
    tensor->lvl2_pos[lvl1_idx + 1] = (lvl1_idx + 1) * dim2_nnz;
  }
  return tensor;
}

// ============================================================================
// CSC tensor utilities
// ============================================================================

struct csc *_allocate_csc(size_t ndim2, size_t dim2_nnz) {
  struct csc *tensor = malloc(sizeof(struct csc));
  tensor->lvl1_size = ndim2; // CSC stores by columns
  tensor->lvl2_pos = malloc((ndim2 + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  tensor->lvl2_nnz = ndim2 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl2_nnz, sizeof(double));
  return tensor;
}

void _free_csc(struct csc *tensor) {
  if (tensor) {
    free(tensor->lvl2_pos);
    free(tensor->lvl2_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void _reset_csc(struct csc *tensor) {
  tensor->lvl2_nnz = 0;
  memset(tensor->lvl2_pos, 0, (tensor->lvl1_size + 1) * sizeof(size_t));
}

struct csc *_generate_csc(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed) {
  srand(seed);

  struct csc *tensor = malloc(sizeof(struct csc));
  tensor->lvl1_size = ndim2; // number of columns
  tensor->lvl2_pos = malloc((ndim2 + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  size_t dim1_nnz = (size_t)(ndim1 * sparsity);
  if (dim1_nnz < 1)
    dim1_nnz = 1;
  if (dim1_nnz > ndim1)
    dim1_nnz = ndim1;

  tensor->lvl2_nnz = ndim2 * dim1_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl2_nnz, sizeof(double));

  for (size_t col = 0; col < ndim2; ++col) {
    // Generate exactly dim1_nnz non-zeros per column
    for (size_t dim1_n = 0; dim1_n < dim1_nnz; ++dim1_n) {
      size_t lvl2_idx = col * dim1_nnz + dim1_n;
      tensor->lvl2_crd[lvl2_idx] = rand_uniform(ndim1); // row index
      tensor->vals[lvl2_idx] = rand_double();
    }
    tensor->lvl2_pos[col + 1] = (col + 1) * dim1_nnz;
  }
  return tensor;
}

// ============================================================================
// COO tensor utilities
// ============================================================================

struct coo *_allocate_coo(size_t nnz) {
  struct coo *tensor = malloc(sizeof(struct coo));
  tensor->lvl1_nnz = nnz;
  tensor->lvl1_crd = malloc(nnz * sizeof(size_t));
  tensor->lvl2_crd = malloc(nnz * sizeof(size_t));
  tensor->vals = calloc(nnz, sizeof(double));
  return tensor;
}

void _free_coo(struct coo *tensor) {
  if (tensor) {
    free(tensor->lvl1_crd);
    free(tensor->lvl2_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void _reset_coo(struct coo *tensor) { tensor->lvl1_nnz = 0; }

struct coo *_generate_coo(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed) {
  srand(seed);

  size_t nnz = (size_t)(ndim1 * ndim2 * sparsity);
  if (nnz < 1)
    nnz = 1;

  struct coo *tensor = malloc(sizeof(struct coo));
  tensor->lvl1_nnz = nnz;
  tensor->lvl1_crd = malloc(nnz * sizeof(size_t));
  tensor->lvl2_crd = malloc(nnz * sizeof(size_t));
  tensor->vals = calloc(nnz, sizeof(double));

  for (size_t idx = 0; idx < nnz; ++idx) {
    tensor->lvl1_crd[idx] = rand_uniform(ndim1);
    tensor->lvl2_crd[idx] = rand_uniform(ndim2);
    tensor->vals[idx] = rand_double();
  }

  return tensor;
}

// ============================================================================
// CSF tensor utilities
// ============================================================================

struct csf *_allocate_csf(size_t ndim1, size_t dim2_nnz, size_t dim3_nnz) {
  struct csf *tensor = malloc(sizeof(struct csf));
  tensor->lvl1_nnz = ndim1;
  tensor->lvl1_crd = malloc(ndim1 * sizeof(size_t));

  tensor->lvl2_nnz = ndim1 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->lvl2_pos = malloc((tensor->lvl2_nnz + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  tensor->lvl3_nnz = tensor->lvl2_nnz * dim3_nnz;
  tensor->lvl3_crd = malloc(tensor->lvl3_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl3_nnz, sizeof(double));
  return tensor;
}

void _free_csf(struct csf *tensor) {
  if (tensor) {
    free(tensor->lvl1_crd);
    free(tensor->lvl2_crd);
    free(tensor->lvl2_pos);
    free(tensor->lvl3_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void _reset_csf(struct csf *tensor) {
  tensor->lvl1_nnz = 0;
  tensor->lvl2_nnz = 0;
  tensor->lvl3_nnz = 0;
  memset(tensor->lvl2_pos, 0, sizeof(size_t));
}

struct csf *_generate_csf(size_t ndim1, size_t ndim2, size_t ndim3, double sparsity, unsigned int seed) {
  srand(seed);

  size_t dim1_nnz = (size_t)(ndim1 * sparsity);
  if (dim1_nnz < 1)
    dim1_nnz = 1;
  if (dim1_nnz > ndim1)
    dim1_nnz = ndim1;

  size_t dim2_nnz = (size_t)(ndim2 * sparsity);
  if (dim2_nnz < 1)
    dim2_nnz = 1;
  if (dim2_nnz > ndim2)
    dim2_nnz = ndim2;

  size_t dim3_nnz = (size_t)(ndim3 * sparsity);
  if (dim3_nnz < 1)
    dim3_nnz = 1;
  if (dim3_nnz > ndim3)
    dim3_nnz = ndim3;

  struct csf *tensor = malloc(sizeof(struct csf));
  tensor->lvl1_nnz = dim1_nnz;
  tensor->lvl1_crd = malloc(dim1_nnz * sizeof(size_t));

  tensor->lvl2_nnz = dim1_nnz * dim2_nnz; // total fibers
  tensor->lvl2_pos = malloc((tensor->lvl2_nnz + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));

  tensor->lvl3_nnz = tensor->lvl2_nnz * dim3_nnz; // total elements
  tensor->lvl3_crd = malloc(tensor->lvl3_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl3_nnz, sizeof(double));

  for (size_t lvl1_idx = 0; lvl1_idx < dim1_nnz; ++lvl1_idx) {
    tensor->lvl1_crd[lvl1_idx] = rand_uniform(ndim1);
    // Generate exactly dim2_nnz fibers per slice
    for (size_t dim2_n = 0; dim2_n < dim2_nnz; ++dim2_n) {
      size_t lvl2_idx = lvl1_idx * dim2_nnz + dim2_n;
      tensor->lvl2_crd[lvl2_idx] = rand_uniform(ndim2);
      // Generate exactly dim3_nnz elements per fiber
      for (size_t dim3_n = 0; dim3_n < dim3_nnz; ++dim3_n) {
        size_t lvl3_idx = lvl2_idx * dim3_nnz + dim3_n;
        tensor->lvl3_crd[lvl3_idx] = rand_uniform(ndim3);
        tensor->vals[lvl3_idx] = rand_double();
      }
      tensor->lvl2_pos[lvl2_idx + 1] = (lvl2_idx + 1) * dim3_nnz;
    }
  }

  return tensor;
}

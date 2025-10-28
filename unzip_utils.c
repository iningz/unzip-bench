#include "unzip_formats.h"
#include <stdlib.h>
#include <string.h>

static inline double rand_double() {
  return (double)rand() / RAND_MAX;
}

static inline size_t rand_size_t() {
  size_t result = 0;
  for (size_t i = 0; i < sizeof(size_t); ++i)
    result = (result << 8) | (rand() & 0xFF);
  return result;
}

static inline size_t rand_uniform(size_t n) {
  return rand_size_t() % n;
}

// Dense tensor utilities
struct dense *allocate_dense(size_t size) {
  struct dense *tensor = malloc(sizeof(struct dense));
  tensor->size = size;
  tensor->vals = calloc(size, sizeof(double));
  return tensor;
}

void free_dense(struct dense *tensor) {
  if (tensor) {
    free(tensor->vals);
    free(tensor);
  }
}

void reset_dense(struct dense *tensor) {
  memset(tensor->vals, 0, tensor->size * sizeof(double));
}

// CSR tensor utilities
struct csr *allocate_csr(size_t ndim1, size_t ndim2, size_t dim2_nnz) {
  struct csr *tensor = malloc(sizeof(struct csr));
  tensor->lvl1_size = ndim1;
  tensor->lvl1_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl1_pos[0] = 0;

  tensor->lvl2_size = ndim2;
  tensor->lvl2_nnz = ndim1 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl2_nnz, sizeof(double));
  return tensor;
}

void free_csr(struct csr *tensor) {
  if (tensor) {
    free(tensor->lvl1_pos);
    free(tensor->lvl2_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void reset_csr(struct csr *tensor) {
  tensor->lvl2_nnz = 0;
  memset(tensor->lvl1_pos, 0, (tensor->lvl1_size + 1) * sizeof(size_t));
}

// CSF tensor utilities
struct csf *allocate_csf(size_t ndim1, size_t ndim2, size_t ndim3, size_t dim2_nnz, size_t dim3_nnz) {
  struct csf *tensor = malloc(sizeof(struct csf));
  tensor->lvl1_size = ndim1;
  tensor->lvl1_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl1_pos[0] = 0;

  tensor->lvl2_size = ndim2;
  tensor->lvl2_nnz = ndim1 * dim2_nnz;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));
  tensor->lvl2_pos = malloc((tensor->lvl2_nnz + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;

  tensor->lvl3_size = ndim3;
  tensor->lvl3_nnz = tensor->lvl2_nnz * dim3_nnz;
  tensor->lvl3_crd = malloc(tensor->lvl3_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl3_nnz, sizeof(double));
  return tensor;
}

void free_csf(struct csf *tensor) {
  if (tensor) {
    free(tensor->lvl1_pos);
    free(tensor->lvl2_crd);
    free(tensor->lvl2_pos);
    free(tensor->lvl3_crd);
    free(tensor->vals);
    free(tensor);
  }
}

void reset_csf(struct csf *tensor) {
  tensor->lvl2_nnz = 0;
  tensor->lvl3_nnz = 0;
  memset(tensor->lvl1_pos, 0, (tensor->lvl1_size + 1) * sizeof(size_t));
  memset(tensor->lvl2_pos, 0, (tensor->lvl1_size + 1) * sizeof(size_t));
}

struct csr *generate_csr(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed) {
  srand(seed);

  struct csr *tensor = malloc(sizeof(struct csr));
  tensor->lvl1_size = ndim1;
  tensor->lvl1_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl1_pos[0] = 0;

  size_t dim2_nnz = (size_t)(ndim2 * sparsity);
  if (dim2_nnz < 1)
    dim2_nnz = 1;
  if (dim2_nnz > ndim2)
    dim2_nnz = ndim2;

  tensor->lvl2_size = ndim2;
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
    tensor->lvl1_pos[lvl1_idx + 1] = (lvl1_idx + 1) * dim2_nnz;
  }
  return tensor;
}

struct csf *generate_csf(size_t ndim1, size_t ndim2, size_t ndim3, double sparsity, unsigned int seed) {
  srand(seed);

  size_t dim2_nnz = ndim2 * sparsity;
  if (dim2_nnz < 1)
    dim2_nnz = 1;
  if (dim2_nnz > ndim2)
    dim2_nnz = ndim2;

  size_t dim3_nnz = ndim3 * sparsity;
  if (dim3_nnz < 1)
    dim3_nnz = 1;
  if (dim3_nnz > ndim3)
    dim3_nnz = ndim3;

  struct csf *tensor = malloc(sizeof(struct csf));
  tensor->lvl1_size = ndim1;
  tensor->lvl1_pos = malloc((ndim1 + 1) * sizeof(size_t));
  tensor->lvl1_pos[0] = 0;

  tensor->lvl2_size = ndim2;
  tensor->lvl2_nnz = ndim1 * dim2_nnz; // total fibers
  tensor->lvl2_pos = malloc((tensor->lvl2_nnz + 1) * sizeof(size_t));
  tensor->lvl2_pos[0] = 0;
  tensor->lvl2_crd = malloc(tensor->lvl2_nnz * sizeof(size_t));

  tensor->lvl3_size = ndim3;
  tensor->lvl3_nnz = tensor->lvl2_nnz * dim3_nnz; // total elements
  tensor->lvl3_crd = malloc(tensor->lvl3_nnz * sizeof(size_t));
  tensor->vals = calloc(tensor->lvl3_nnz, sizeof(double));

  for (size_t lvl1_idx = 0; lvl1_idx < ndim1; ++lvl1_idx) {
    // Generate exactly lvl2_fiber_nnz fibers per slice
    for (size_t dim2_n = 0; dim2_n < dim2_nnz; ++dim2_n) {
      size_t lvl2_idx = lvl1_idx * dim2_nnz + dim2_n;
      tensor->lvl2_crd[lvl2_idx] = rand_uniform(ndim2);
      // Generate exactly lvl3_row_nnz elements per fiber
      for (size_t dim3_n = 0; dim3_n < dim3_nnz; ++dim3_n) {
        size_t lvl3_idx = lvl2_idx * dim3_nnz + dim3_n;
        tensor->lvl3_crd[lvl3_idx] = rand_uniform(ndim3);
        tensor->vals[lvl3_idx] = rand_double();
      }
      tensor->lvl2_pos[lvl2_idx + 1] = (lvl2_idx + 1) * dim3_nnz;
    }
    tensor->lvl1_pos[lvl1_idx + 1] = (lvl1_idx + 1) * dim2_nnz;
  }

  return tensor;
}
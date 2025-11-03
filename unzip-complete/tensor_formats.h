#ifndef FORMATS_H
#define FORMATS_H

#include <stddef.h>

// 1D Dense Vector
struct dense {
  size_t lvl1_size; // Dense

  double *vals; // values (size: lvl1_size)
};

// 2D Compressed Sparse Row (CSR) format
struct csr {
  // Level 1: Dense
  size_t lvl1_size; // size: number of rows

  // Level 2: Compressed
  size_t *lvl2_pos; // size: lvl1_size + 1
  size_t lvl2_nnz;
  size_t *lvl2_crd; // size: lvl2_nnz

  double *vals; // size: lvl2_nnz
};

// 2D Compressed Sparse Column (CSC) format
struct csc {
  // Level 1: Dense
  size_t lvl1_size; // size: number of columns

  // Level 2: Compressed
  size_t *lvl2_pos; // size: lvl1_size + 1
  size_t lvl2_nnz;
  size_t *lvl2_crd; // size: lvl2_nnz

  double *vals; // size: lvl2_nnz
};

// 3D Compressed Sparse Fiber (CSF) format
struct csf {
  // Level 1: Comrpessed
  size_t lvl1_nnz;
  size_t *lvl1_crd; // size: lvl1_nnz

  // Level 2: Compressed
  size_t *lvl2_pos; // size: lvl1_nnz + 1
  size_t lvl2_nnz;
  size_t *lvl2_crd; // size: lvl2_nnz

  // Level 3: Comrpessed
  size_t *lvl3_pos; // size: lvl2_nnz + 1
  size_t lvl3_nnz;
  size_t *lvl3_crd; // size: lvl3_nnz

  double *vals; // size: lvl3_nnz
};

// 2D Coordinate (COO) format
struct coo {
  // Level 1: Compressed (non-unique)
  size_t lvl1_nnz;
  size_t *lvl1_crd; // size: lvl1_nnz

  // Level 2: Singleton
  size_t *lvl2_crd; // size: lvl1_nnz

  double *vals;
};

// Generic dispatch macros

#define allocate_tensor(FMT, ...) _allocate_##FMT(__VA_ARGS__)
#define generate_tensor(FMT, ...) _generate_##FMT(__VA_ARGS__)

#define reset_tensor(T)                                                                                                \
  _Generic((T),                                                                                                        \
      struct dense *: _reset_dense,                                                                                    \
      struct csr *: _reset_csr,                                                                                        \
      struct csc *: _reset_csc,                                                                                        \
      struct coo *: _reset_coo,                                                                                        \
      struct csf *: _reset_csf)(T)

#define free_tensor(T)                                                                                                 \
  _Generic((T),                                                                                                        \
      struct dense *: _free_dense,                                                                                     \
      struct csr *: _free_csr,                                                                                         \
      struct csc *: _free_csc,                                                                                         \
      struct coo *: _free_coo,                                                                                         \
      struct csf *: _free_csf)(T)

// Internal utility function declarations (use generic macros below instead)

// Dense utilities
struct dense *_allocate_dense(size_t n);
void _free_dense(struct dense *tensor);
void _reset_dense(struct dense *tensor);
struct dense *_generate_dense(size_t n, unsigned int seed);

// CSR utilities
struct csr *_allocate_csr(size_t ndim1, size_t dim2_nnz);
void _free_csr(struct csr *tensor);
void _reset_csr(struct csr *tensor);
struct csr *_generate_csr(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed);

// CSC utilities
struct csc *_allocate_csc(size_t ndim2, size_t dim2_nnz);
void _free_csc(struct csc *tensor);
void _reset_csc(struct csc *tensor);
struct csc *_generate_csc(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed);

// COO utilities
struct coo *_allocate_coo(size_t nnz);
void _free_coo(struct coo *tensor);
void _reset_coo(struct coo *tensor);
struct coo *_generate_coo(size_t ndim1, size_t ndim2, double sparsity, unsigned int seed);

// CSF utilities
struct csf *_allocate_csf(size_t ndim1, size_t dim2_nnz, size_t dim3_nnz);
void _free_csf(struct csf *tensor);
void _reset_csf(struct csf *tensor);
struct csf *_generate_csf(size_t ndim1, size_t ndim2, size_t ndim3, double sparsity, unsigned int seed);

#endif /* FORMATS_H */

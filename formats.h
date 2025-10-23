#ifndef FORMATS_H
#define FORMATS_H

#include <stddef.h>

// 1D Dense Vector
struct dense {
  size_t size;  // size of the vector
  double *vals; // values (size: size)
};

// 2D Sparse Matrix in CSR (Compressed Sparse Row) format
struct csr {
  // Level 1: rows
  size_t lvl1_size; // size of dimension 1 (rows)
  size_t *lvl1_pos; // position array (size: lvl1_size + 1)

  // Level 2: columns
  size_t lvl2_size; // size of dimension 2 (cols)
  size_t lvl2_nnz;  // number of non-zero elements
  size_t *lvl2_crd; // coordinate array (size: lvl2_nnz)

  double *vals; // values (size: lvl2_nnz)
};

// 3D Sparse Tensor in CSF (Compressed Sparse Fiber) format
struct csf {
  // Level 1: dim1 dimension
  size_t lvl1_size; // size of dimension 1 (rows)
  size_t *lvl1_pos; // position array (size: lvl1_size + 1)

  // Level 2: dim2 dimension within each dim1
  size_t lvl2_size; // size of dimension 2
  size_t lvl2_nnz;  // number of non-zero (dim1,dim2) fibers
  size_t *lvl2_pos; // position array (size: lvl2_nnz + 1)
  size_t *lvl2_crd; // coordinate array (size: lvl2_nnz)

  // Level 3: dim3 dimension within each (dim1,dim2)
  size_t lvl3_size; // size of dimension 3
  size_t lvl3_nnz;  // number of non-zero elements
  size_t *lvl3_crd; // coordinate array (size: lvl3_nnz)

  double *vals; // values (size: lvl3_nnz)
};

#endif /* FORMATS_H */

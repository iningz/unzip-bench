#ifndef UNZIP_KERNELS_H
#define UNZIP_KERNELS_H

#include "unzip_formats.h"

/* Kernel function declarations */

/* A(i,j) = B(i,j) * C(j,i) - Hadamard product with transpose */
void hadamard_transpose(struct csr *t1, struct csr *t2, struct csr *res);

/* A(i, j) = B(i, k) * C(k, j) - Matrix multiplication */
void matmul(struct csr *t1, struct csr *t2, struct csr *res);

/* A(i, j) = B(i, k) * C(k, j) * D(k, j) - Matrix multiplication with Hadamard */
void matmul_hadamard(struct csr *t1, struct csr *t2, struct csr *t3, struct csr *res);

/* y(i) = B(i, j) * C(j, i) - Hadamard transpose with reduction to vector */
void hadamard_transpose_reduce(struct csr *t1, struct csr *t2, struct dense *res);

/* y(i) = B(i, j, k) * C(i, k, j) - 3D tensor contraction with permutation */
void permute_contract(struct csf *t1, struct csf *t2, struct dense *res);

#endif /* KERNELS_H */

// Utility functions for CSR matrix operations

#include "utils.h"
#include <stdlib.h>

struct csr *generate_csr_matrix(int m, int n, double sparsity,
                                unsigned int seed) {
  srand(seed);

  int estimated_nnz = (int)(m * n * sparsity);
  if (estimated_nnz < 1)
    estimated_nnz = 1;

  struct csr *mat = (struct csr *)malloc(sizeof(struct csr));
  mat->dim1_size = m;
  mat->vals = (double *)calloc(estimated_nnz * 2, sizeof(double));
  mat->crd = (int *)calloc(estimated_nnz * 2, sizeof(int));
  mat->pos = (int *)calloc(m + 1, sizeof(int));

  mat->nnz = 0;
  mat->pos[0] = 0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if ((double)rand() / RAND_MAX < sparsity) {
        mat->vals[mat->nnz] = (double)rand() / RAND_MAX;
        mat->crd[mat->nnz] = j;
        mat->nnz++;
      }
    }
    mat->pos[i + 1] = mat->nnz;
  }

  return mat;
}

struct csr *allocate_result_csr(int m, int max_nnz) {
  struct csr *mat = (struct csr *)malloc(sizeof(struct csr));
  mat->dim1_size = m;
  mat->nnz = 0;
  mat->vals = (double *)calloc(max_nnz, sizeof(double));
  mat->crd = (int *)calloc(max_nnz, sizeof(int));
  mat->pos = (int *)calloc(m + 1, sizeof(int));
  return mat;
}

void reset_csr_result(struct csr *mat) {
  mat->nnz = 0;
  for (int i = 0; i <= mat->dim1_size; i++) {
    mat->pos[i] = 0;
  }
}

void free_csr_matrix(struct csr *mat) {
  if (mat) {
    free(mat->vals);
    free(mat->crd);
    free(mat->pos);
    free(mat);
  }
}

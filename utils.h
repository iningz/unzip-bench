// Utility functions for CSR matrix operations

#ifndef UTILS_H
#define UTILS_H

struct csr {
  int dim1_size;
  int nnz;
  double *vals;
  int *crd;
  int *pos;
};

// Function declarations
struct csr *generate_csr_matrix(int m, int n, double sparsity,
                                unsigned int seed);
struct csr *allocate_result_csr(int m, int max_nnz);
void reset_csr_result(struct csr *mat);
void free_csr_matrix(struct csr *mat);

#endif // UTILS_H

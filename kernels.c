// Kernel implementation: A(i,j) = B(i,j) * C(j,i)
// Hadamard product with transpose pattern

#include "utils.h"
#include <stdio.h>

void hadamard_transpose(struct csr *m1, struct csr *m2, struct csr *res) {
  for (int i = 0; i < m1->dim1_size; i++) {
    // Traverse only non-zero elements in M1's row i
    for (int m1_i = m1->pos[i]; m1_i < m1->pos[i + 1]; m1_i++) {
      int j = m1->crd[m1_i];
      double m1_val = m1->vals[m1_i];

      // Locate element at position (j, i) in M2
      for (int m2_i = m2->pos[j]; m2_i < m2->pos[j + 1]; m2_i++) {
        int k = m2->crd[m2_i];
        if (k == i) {
          double m2_val = m2->vals[m2_i];
          if (m1_val != 0.0 && m2_val != 0.0) {
            res->vals[res->nnz] = m1_val * m2_val;
            res->crd[res->nnz] = j;
            ++res->nnz;
          }
        }
      }
    }
    res->pos[i + 1] = res->nnz;
  }
}

void test_hadamard_transpose() {
  // Hand-write two 2x2 matrices in CSR format
  // Matrix B = [1 2; 0 3]
  // Matrix C = [4 0; 5 6]
  struct csr B, C, A;
  double b_vals[3] = {1, 2, 3};
  int b_crd[3] = {0, 1, 1};
  int b_pos[3] = {0, 2, 3};
  B.vals = b_vals;
  B.crd = b_crd;
  B.pos = b_pos;
  B.dim1_size = 2;
  B.nnz = 3;

  double c_vals[3] = {4, 5, 6};
  int c_crd[3] = {0, 0, 1};
  int c_pos[3] = {0, 1, 3};
  C.vals = c_vals;
  C.crd = c_crd;
  C.pos = c_pos;
  C.dim1_size = 2;
  C.nnz = 3;

  double res_vals[4] = {0};
  int res_crd[4] = {0};
  int res_pos[3] = {0};
  A.vals = res_vals;
  A.crd = res_crd;
  A.pos = res_pos;
  A.dim1_size = 2;
  A.nnz = 0;

  hadamard_transpose(&B, &C, &A);

  for (int i = 0; i < A.dim1_size; i++) {
    int col_idx = 0;
    for (int j = A.pos[i]; j < A.pos[i + 1]; j++) {
      // Print zeros for missing columns
      while (col_idx < A.crd[j]) {
        printf(" %.1f", 0.0);
        col_idx++;
      }
      // Print the non-zero value
      printf(" %.1f", A.vals[j]);
      col_idx = A.crd[j] + 1;
    }
    // Print remaining zeros in the row
    while (col_idx < A.dim1_size) {
      printf(" %.1f", 0.0);
      col_idx++;
    }
    printf("\n");
  }
}

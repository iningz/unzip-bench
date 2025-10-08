// Kernel implementation: A(i,j) = B(i,j) * C(j,i)
// Hadamard product with transpose pattern

#include "utils.h"

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
            res->crd[res->nnz] = k;
            ++res->nnz;
          }
        }
      }
    }
    res->pos[i + 1] = res->nnz;
  }
}

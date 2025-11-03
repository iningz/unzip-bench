#include "unzip_kernels.h"
#include <stdio.h>
#include <stdlib.h>

static void print_csr(struct csr *tensor) {
  for (int row_idx = 0; row_idx < tensor->lvl1_size; row_idx++) {
    int col_idx = 0;
    for (int nnz_idx = tensor->lvl1_pos[row_idx]; nnz_idx < tensor->lvl1_pos[row_idx + 1];
         nnz_idx++) {
      while (col_idx < tensor->lvl2_crd[nnz_idx]) {
        printf(" %.1f", 0.0);
        col_idx++;
      }
      printf(" %.1f", tensor->vals[nnz_idx]);
      col_idx = tensor->lvl2_crd[nnz_idx] + 1;
    }
    while (col_idx < tensor->lvl1_size) {
      printf(" %.1f", 0.0);
      col_idx++;
    }
    printf("\n");
  }
}

static void print_dense(struct dense *tensor) {
  for (int i = 0; i < tensor->size; i++) {
    printf("y[%d] = %.1f\n", i, tensor->vals[i]);
  }
}

void test_hadamard_transpose() {
  // B = [1 2; 0 3]
  struct csr B;
  double b_vals[3] = {1, 2, 3};
  size_t b_lvl2_crd[3] = {0, 1, 1};
  size_t b_lvl1_pos[3] = {0, 2, 3};
  B.vals = b_vals;
  B.lvl2_crd = b_lvl2_crd;
  B.lvl1_pos = b_lvl1_pos;
  B.lvl1_size = 2;
  B.lvl2_size = 2;
  B.lvl2_nnz = 3;

  // C = [4 0; 5 6]
  struct csr C;
  double c_vals[3] = {4, 5, 6};
  size_t c_lvl2_crd[3] = {0, 0, 1};
  size_t c_lvl1_pos[3] = {0, 1, 3};
  C.vals = c_vals;
  C.lvl2_crd = c_lvl2_crd;
  C.lvl1_pos = c_lvl1_pos;
  C.lvl1_size = 2;
  C.lvl2_size = 2;
  C.lvl2_nnz = 3;

  // A(i,j) = B(i,j) * C(j,i)
  // Expected: A = [4 10; 0 18]
  struct csr A;
  double res_vals[4] = {0};
  size_t res_lvl2_crd[4] = {0};
  size_t res_lvl1_pos[3] = {0};
  A.vals = res_vals;
  A.lvl2_crd = res_lvl2_crd;
  A.lvl1_pos = res_lvl1_pos;
  A.lvl1_size = 2;
  A.lvl2_size = 2;
  A.lvl2_nnz = 0;

  hadamard_transpose(&B, &C, &A);
  print_csr(&A);
}

void test_matmul() {
  // B = [1 1; 0 0]
  struct csr B;
  double b_vals[2] = {1, 1};
  size_t b_lvl2_crd[2] = {0, 1};
  size_t b_lvl1_pos[3] = {0, 2, 2};
  B.vals = b_vals;
  B.lvl2_crd = b_lvl2_crd;
  B.lvl1_pos = b_lvl1_pos;
  B.lvl1_size = 2;
  B.lvl2_size = 2;
  B.lvl2_nnz = 2;

  // C = [1 0; 1 0]
  struct csr C;
  double c_vals[2] = {1, 1};
  size_t c_lvl2_crd[2] = {0, 0};
  size_t c_lvl1_pos[3] = {0, 1, 2};
  C.vals = c_vals;
  C.lvl2_crd = c_lvl2_crd;
  C.lvl1_pos = c_lvl1_pos;
  C.lvl1_size = 2;
  C.lvl2_size = 2;
  C.lvl2_nnz = 2;

  // A(i,j) = B(i,k) * C(k,j)
  // Expected: A = [2 0; 0 0]
  struct csr A;
  double res_vals[10] = {0};
  size_t res_lvl2_crd[10] = {0};
  size_t res_lvl1_pos[3] = {0};
  A.vals = res_vals;
  A.lvl2_crd = res_lvl2_crd;
  A.lvl1_pos = res_lvl1_pos;
  A.lvl1_size = 2;
  A.lvl2_size = 2;
  A.lvl2_nnz = 0;

  matmul(&B, &C, &A);
  print_csr(&A);
}

void test_matmul_hadamard() {
  // B = [1 2; 0 3]
  struct csr B;
  double b_vals[3] = {1, 2, 3};
  size_t b_lvl2_crd[3] = {0, 1, 1};
  size_t b_lvl1_pos[3] = {0, 2, 3};
  B.vals = b_vals;
  B.lvl2_crd = b_lvl2_crd;
  B.lvl1_pos = b_lvl1_pos;
  B.lvl1_size = 2;
  B.lvl2_size = 2;
  B.lvl2_nnz = 3;

  // C = [1 0; 0 1]
  struct csr C;
  double c_vals[2] = {1, 1};
  size_t c_lvl2_crd[2] = {0, 1};
  size_t c_lvl1_pos[3] = {0, 1, 2};
  C.vals = c_vals;
  C.lvl2_crd = c_lvl2_crd;
  C.lvl1_pos = c_lvl1_pos;
  C.lvl1_size = 2;
  C.lvl2_size = 2;
  C.lvl2_nnz = 2;

  // D = [2 0; 0 2]
  struct csr D;
  double d_vals[2] = {2, 2};
  size_t d_lvl2_crd[2] = {0, 1};
  size_t d_lvl1_pos[3] = {0, 1, 2};
  D.vals = d_vals;
  D.lvl2_crd = d_lvl2_crd;
  D.lvl1_pos = d_lvl1_pos;
  D.lvl1_size = 2;
  D.lvl2_size = 2;
  D.lvl2_nnz = 2;

  // A(i,j) = B(i,k) * C(k,j) * D(k,j)
  // Expected: A = [2 0; 0 6]
  struct csr A;
  double res_vals[10] = {0};
  size_t res_lvl2_crd[10] = {0};
  size_t res_lvl1_pos[3] = {0};
  A.vals = res_vals;
  A.lvl2_crd = res_lvl2_crd;
  A.lvl1_pos = res_lvl1_pos;
  A.lvl1_size = 2;
  A.lvl2_size = 2;
  A.lvl2_nnz = 0;

  matmul_hadamard(&B, &C, &D, &A);
  print_csr(&A);
}

void test_hadamard_transpose_reduce() {
  // B = [1 2; 0 3]
  struct csr B;
  double b_vals[3] = {1, 2, 3};
  size_t b_lvl2_crd[3] = {0, 1, 1};
  size_t b_lvl1_pos[3] = {0, 2, 3};
  B.vals = b_vals;
  B.lvl2_crd = b_lvl2_crd;
  B.lvl1_pos = b_lvl1_pos;
  B.lvl1_size = 2;
  B.lvl2_size = 2;
  B.lvl2_nnz = 3;

  // C = [4 0; 5 6]
  struct csr C;
  double c_vals[3] = {4, 5, 6};
  size_t c_lvl2_crd[3] = {0, 0, 1};
  size_t c_lvl1_pos[3] = {0, 1, 3};
  C.vals = c_vals;
  C.lvl2_crd = c_lvl2_crd;
  C.lvl1_pos = c_lvl1_pos;
  C.lvl1_size = 2;
  C.lvl2_size = 2;
  C.lvl2_nnz = 3;

  // y(i) = sum_j B(i,j) * C(j,i)
  // Expected: y = [14, 18]
  struct dense y;
  double res_vals[2] = {0};
  y.vals = res_vals;
  y.size = 2;

  hadamard_transpose_reduce(&B, &C, &y);
  print_dense(&y);
}

void test_permute_contract() {
  // B has non-zeros at: B[0,0,0]=1, B[0,0,1]=2, B[1,1,1]=3
  struct csf B;
  double b_vals[3] = {1, 2, 3};
  size_t b_lvl3_crd[3] = {0, 1, 1};
  size_t b_lvl2_crd[3] = {0, 0, 1};
  size_t b_lvl2_pos[3] = {0, 2, 3};
  size_t b_lvl1_pos[3] = {0, 1, 2};
  B.vals = b_vals;
  B.lvl3_crd = b_lvl3_crd;
  B.lvl2_crd = b_lvl2_crd;
  B.lvl2_pos = b_lvl2_pos;
  B.lvl1_pos = b_lvl1_pos;
  B.lvl1_size = 2;
  B.lvl2_size = 2;
  B.lvl3_size = 2;
  B.lvl2_nnz = 2;
  B.lvl3_nnz = 3;

  // C has non-zeros at: C[0,0,0]=4, C[1,0,1]=5, C[1,1,1]=6
  struct csf C;
  double c_vals[3] = {4, 5, 6};
  size_t c_lvl3_crd[3] = {0, 1, 1};
  size_t c_lvl2_crd[3] = {0, 0, 1};
  size_t c_lvl2_pos[3] = {0, 1, 3};
  size_t c_lvl1_pos[3] = {0, 1, 2};
  C.vals = c_vals;
  C.lvl3_crd = c_lvl3_crd;
  C.lvl2_crd = c_lvl2_crd;
  C.lvl2_pos = c_lvl2_pos;
  C.lvl1_pos = c_lvl1_pos;
  C.lvl1_size = 2;
  C.lvl2_size = 2;
  C.lvl3_size = 2;
  C.lvl2_nnz = 2;
  C.lvl3_nnz = 3;

  // y(i) = sum_jk B(i,j,k) * C(i,k,j)
  // Expected: y = [4, 18]
  struct dense y;
  double res_vals[2] = {0};
  y.vals = res_vals;
  y.size = 2;

  permute_contract(&B, &C, &y);
  print_dense(&y);
}

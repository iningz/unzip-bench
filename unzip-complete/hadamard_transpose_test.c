#include "hadamard_transpose.h"
#include "tensor_formats.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to create a simple test CSR matrix for B
#if defined(FORMAT_B_CSR) || defined(FORMAT_C_CSR)
static struct csr *create_test_csr_b() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  struct csr *B = allocate_csr(3, 3);
  B->lvl2_nnz = 5;
  // Row 0: (0,0)=1.0, (0,2)=2.0
  B->lvl2_pos[0] = 0;
  B->lvl2_crd[0] = 0;
  B->vals[0] = 1.0;
  B->lvl2_crd[1] = 2;
  B->vals[1] = 2.0;
  B->lvl2_pos[1] = 2;
  // Row 1: (1,1)=3.0
  B->lvl2_crd[2] = 1;
  B->vals[2] = 3.0;
  B->lvl2_pos[2] = 3;
  // Row 2: (2,0)=4.0, (2,2)=5.0
  B->lvl2_crd[3] = 0;
  B->vals[3] = 4.0;
  B->lvl2_crd[4] = 2;
  B->vals[4] = 5.0;
  B->lvl2_pos[3] = 5;
  return B;
}
#endif

#if defined(FORMAT_C_CSR)
static struct csr *create_test_csr_c() {
  // C(j,i) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // Same as B for simplicity
  return create_test_csr_b();
}
#endif

#if defined(FORMAT_B_CSC) || defined(FORMAT_C_CSC)
static struct csc *create_test_csc_b() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // In CSC: column 0: (0,0)=1.0, (2,0)=4.0
  //         column 1: (1,1)=3.0
  //         column 2: (0,2)=2.0, (2,2)=5.0
  struct csc *B = allocate_csc(3, 3);
  B->lvl1_size = 3; // 3 columns
  B->lvl2_nnz = 5;
  // Column 0
  B->lvl2_pos[0] = 0;
  B->lvl2_crd[0] = 0; // row 0
  B->vals[0] = 1.0;
  B->lvl2_crd[1] = 2; // row 2
  B->vals[1] = 4.0;
  B->lvl2_pos[1] = 2;
  // Column 1
  B->lvl2_crd[2] = 1; // row 1
  B->vals[2] = 3.0;
  B->lvl2_pos[2] = 3;
  // Column 2
  B->lvl2_crd[3] = 0; // row 0
  B->vals[3] = 2.0;
  B->lvl2_crd[4] = 2; // row 2
  B->vals[4] = 5.0;
  B->lvl2_pos[3] = 5;
  return B;
}
#endif

#if defined(FORMAT_C_CSC)
static struct csc *create_test_csc_c() {
  // C(j,i) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // Same as B for simplicity
  return create_test_csc_b();
}
#endif

#if defined(FORMAT_B_COO) || defined(FORMAT_C_COO)
static struct coo *create_test_coo_b() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  struct coo *B = allocate_coo(5);
  B->lvl1_nnz = 5;
  B->lvl1_crd[0] = 0;
  B->lvl2_crd[0] = 0;
  B->vals[0] = 1.0;
  B->lvl1_crd[1] = 0;
  B->lvl2_crd[1] = 2;
  B->vals[1] = 2.0;
  B->lvl1_crd[2] = 1;
  B->lvl2_crd[2] = 1;
  B->vals[2] = 3.0;
  B->lvl1_crd[3] = 2;
  B->lvl2_crd[3] = 0;
  B->vals[3] = 4.0;
  B->lvl1_crd[4] = 2;
  B->lvl2_crd[4] = 2;
  B->vals[4] = 5.0;
  return B;
}
#endif

#if defined(FORMAT_C_COO)
static struct coo *create_test_coo_c() {
  // C(j,i) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // Same as B for simplicity
  return create_test_coo_b();
}
#endif

#if defined(FORMAT_A_CSR)
static int verify_result_csr(struct csr *A, const char *test_name) {
  // Verify A(i,j) = B(i,j) * C(j,i) in CSR format
  // Expected non-zeros:
  // A(0,0) = B(0,0) * C(0,0) = 1.0 * 1.0 = 1.0
  // A(0,2) = B(0,2) * C(2,0) = 2.0 * 4.0 = 8.0
  // A(1,1) = B(1,1) * C(1,1) = 3.0 * 3.0 = 9.0
  // A(2,0) = B(2,0) * C(0,2) = 4.0 * 2.0 = 8.0
  // A(2,2) = B(2,2) * C(2,2) = 5.0 * 5.0 = 25.0

  double expected_vals[5] = {1.0, 8.0, 9.0, 8.0, 25.0};
  size_t expected_cols[5] = {0, 2, 1, 0, 2};
  size_t expected_pos[4] = {0, 2, 3, 5};

  int passed = 1;

  if (A->lvl2_nnz != 5) {
    printf("  FAIL %s: Expected 5 non-zeros, got %zu\n", test_name, A->lvl2_nnz);
    passed = 0;
  }

  for (size_t i = 0; i < 3; i++) {
    if (A->lvl2_pos[i] != expected_pos[i]) {
      printf("  FAIL %s: Row %zu pos mismatch: expected %zu, got %zu\n", test_name, i, expected_pos[i], A->lvl2_pos[i]);
      passed = 0;
    }
  }

  for (size_t idx = 0; idx < 5; idx++) {
    if (A->lvl2_crd[idx] != expected_cols[idx]) {
      printf("  FAIL %s: Entry %zu col mismatch: expected %zu, got %zu\n", test_name, idx, expected_cols[idx],
             A->lvl2_crd[idx]);
      passed = 0;
    }
    if (fabs(A->vals[idx] - expected_vals[idx]) > 1e-9) {
      printf("  FAIL %s: Entry %zu val mismatch: expected %.1f, got %.1f\n", test_name, idx, expected_vals[idx],
             A->vals[idx]);
      passed = 0;
    }
  }

  if (passed) {
    printf("  PASS %s\n", test_name);
  }
  return passed;
}
#endif

#if defined(FORMAT_A_CSC)
static int verify_result_csc(struct csc *A, const char *test_name) {
  // Verify A(i,j) = B(i,j) * C(j,i) in CSC format (stored by columns)
  // Expected: Column 0: A(0,0)=1.0, A(2,0)=8.0
  //           Column 1: A(1,1)=9.0
  //           Column 2: A(0,2)=8.0, A(2,2)=25.0

  double expected_vals[5] = {1.0, 8.0, 9.0, 8.0, 25.0};
  size_t expected_rows[5] = {0, 2, 1, 0, 2};
  size_t expected_pos[4] = {0, 2, 3, 5};

  int passed = 1;

  if (A->lvl2_nnz != 5) {
    printf("  FAIL %s: Expected 5 non-zeros, got %zu\n", test_name, A->lvl2_nnz);
    passed = 0;
  }

  for (size_t j = 0; j < 3; j++) {
    if (A->lvl2_pos[j] != expected_pos[j]) {
      printf("  FAIL %s: Col %zu pos mismatch: expected %zu, got %zu\n", test_name, j, expected_pos[j], A->lvl2_pos[j]);
      passed = 0;
    }
  }

  for (size_t idx = 0; idx < 5; idx++) {
    if (A->lvl2_crd[idx] != expected_rows[idx]) {
      printf("  FAIL %s: Entry %zu row mismatch: expected %zu, got %zu\n", test_name, idx, expected_rows[idx],
             A->lvl2_crd[idx]);
      passed = 0;
    }
    if (fabs(A->vals[idx] - expected_vals[idx]) > 1e-9) {
      printf("  FAIL %s: Entry %zu val mismatch: expected %.1f, got %.1f\n", test_name, idx, expected_vals[idx],
             A->vals[idx]);
      passed = 0;
    }
  }

  if (passed) {
    printf("  PASS %s\n", test_name);
  }
  return passed;
}
#endif

int main() {
  int passed = 0;

  printf("Running Hadamard Transpose Test\n");
  printf("================================\n");

  // Determine configuration from compile-time flags
#if defined(FORMAT_A_CSR)
  const char *a_fmt = "CSR";
#elif defined(FORMAT_A_CSC)
  const char *a_fmt = "CSC";
#elif defined(FORMAT_A_COO)
  const char *a_fmt = "COO";
#else
  const char *a_fmt = "UNDEFINED";
#endif

#if defined(FORMAT_B_CSR)
  const char *b_fmt = "CSR";
#elif defined(FORMAT_B_CSC)
  const char *b_fmt = "CSC";
#elif defined(FORMAT_B_COO)
  const char *b_fmt = "COO";
#else
  const char *b_fmt = "UNDEFINED";
#endif

#if defined(FORMAT_C_CSR)
  const char *c_fmt = "CSR";
#elif defined(FORMAT_C_CSC)
  const char *c_fmt = "CSC";
#elif defined(FORMAT_C_COO)
  const char *c_fmt = "COO";
#else
  const char *c_fmt = "UNDEFINED";
#endif

#if defined(SEARCH_B)
  const char *search = "B";
#elif defined(SEARCH_C)
  const char *search = "C";
#else
  const char *search = "UNDEFINED";
#endif

  printf("Configuration: A=%s, B=%s, C=%s, SEARCH=%s\n\n", a_fmt, b_fmt, c_fmt, search);

  // Allocate and run test based on compile-time configuration
#if defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSR)
  struct csr *A = allocate_csr(3, 5);
  struct csr *B = create_test_csr_b();
  struct csr *C = create_test_csr_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-csr-csr");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSC)
  struct csr *A = allocate_csr(3, 5);
  struct csr *B = create_test_csr_b();
  struct csc *C = create_test_csc_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-csr-csc");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_COO)
  struct csr *A = allocate_csr(3, 5);
  struct csr *B = create_test_csr_b();
  struct coo *C = create_test_coo_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-csr-coo");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSR)
  struct csr *A = allocate_csr(3, 5);
  struct coo *B = create_test_coo_b();
  struct csr *C = create_test_csr_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-coo-csr");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSC)
  struct csr *A = allocate_csr(3, 5);
  struct coo *B = create_test_coo_b();
  struct csc *C = create_test_csc_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-coo-csc");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_COO)
  struct csr *A = allocate_csr(3, 5);
  struct coo *B = create_test_coo_b();
  struct coo *C = create_test_coo_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csr(A, "csr-coo-coo");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSR)
  struct csc *A = allocate_csc(3, 5);
  struct csc *B = create_test_csc_b();
  struct csr *C = create_test_csr_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csc(A, "csc-csc-csr");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSC)
  struct csc *A = allocate_csc(3, 5);
  struct csc *B = create_test_csc_b();
  struct csc *C = create_test_csc_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csc(A, "csc-csc-csc");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_COO)
  struct csc *A = allocate_csc(3, 5);
  struct csc *B = create_test_csc_b();
  struct coo *C = create_test_coo_c();
  reset_tensor(A);
  hadamard_transpose(A, B, C);
  passed = verify_result_csc(A, "csc-csc-coo");
  free_tensor(A);
  free_tensor(B);
  free_tensor(C);

#else
  printf("ERROR: Unsupported or missing format configuration\n");
  return 1;
#endif

  printf("\n================================\n");
  printf("Test Result: %s\n", passed ? "PASSED" : "FAILED");

  return passed ? 0 : 1;
}

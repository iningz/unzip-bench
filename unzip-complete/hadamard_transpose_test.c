#include "hadamard_transpose.h"
#include "tensor_formats.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper to create a simple test CSR matrix
struct csr *create_test_csr_B() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  struct csr *B = allocate_tensor(csr, 3, 3);
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

struct csr *create_test_csr_C() {
  // C(j,i) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // Same as B for simplicity
  return create_test_csr_B();
}

struct csc *create_test_csc_B() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  // In CSC: column 0: (0,0)=1.0, (2,0)=4.0
  //         column 1: (1,1)=3.0
  //         column 2: (0,2)=2.0, (2,2)=5.0
  struct csc *B = allocate_tensor(csc, 3, 3);
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

struct csc *create_test_csc_C() { return create_test_csc_B(); }

struct coo *create_test_coo_B() {
  // B(i,j) = [[1.0, 0, 2.0], [0, 3.0, 0], [4.0, 0, 5.0]]
  struct coo *B = allocate_tensor(coo, 5);
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

struct coo *create_test_coo_C() { return create_test_coo_B(); }

void print_csr(const char *name, struct csr *mat) {
  printf("%s (CSR):\n", name);
  printf("  lvl1_size=%zu, lvl2_nnz=%zu\n", mat->lvl1_size, mat->lvl2_nnz);
  for (size_t i = 0; i < mat->lvl1_size; i++) {
    printf("  Row %zu: ", i);
    for (size_t pos = mat->lvl2_pos[i]; pos < mat->lvl2_pos[i + 1]; pos++) {
      printf("(%zu)=%.1f ", mat->lvl2_crd[pos], mat->vals[pos]);
    }
    printf("\n");
  }
}

void print_csc(const char *name, struct csc *mat) {
  printf("%s (CSC):\n", name);
  printf("  lvl1_size=%zu, lvl2_nnz=%zu\n", mat->lvl1_size, mat->lvl2_nnz);
  for (size_t j = 0; j < mat->lvl1_size; j++) {
    printf("  Col %zu: ", j);
    for (size_t pos = mat->lvl2_pos[j]; pos < mat->lvl2_pos[j + 1]; pos++) {
      printf("(%zu)=%.1f ", mat->lvl2_crd[pos], mat->vals[pos]);
    }
    printf("\n");
  }
}

int verify_result_csr(struct csr *A, const char *test_name) {
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

int verify_result_csc(struct csc *A, const char *test_name) {
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

int main() {
  int total_tests = 0;
  int passed_tests = 0;

  printf("Running Hadamard Transpose Tests\n");
  printf("=================================\n\n");

  // Generate test data for all format combinations of B and C
  printf("Generating test data...\n");
  struct csr *B_csr = create_test_csr_B();
  struct csr *C_csr = create_test_csr_C();
  struct csc *B_csc = create_test_csc_B();
  struct csc *C_csc = create_test_csc_C();
  struct coo *B_coo = create_test_coo_B();
  struct coo *C_coo = create_test_coo_C();

  // Allocate output tensors once
  struct csr *A_csr = allocate_tensor(csr, 3, 5);
  struct csc *A_csc = allocate_tensor(csc, 3, 5);
  printf("Test data generated.\n\n");

  // Test 1: csr-csr-csr-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_csr, C_csr, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-csr-csr-snd"))
      passed_tests++;
  }

  // Test 2: csr-csr-csc-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_csr, C_csc, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-csr-csc-snd"))
      passed_tests++;
  }

  // Test 3: csr-csr-coo-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_csr, C_coo, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-csr-coo-snd"))
      passed_tests++;
  }

  // Test 4: csr-coo-csr-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_coo, C_csr, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-coo-csr-snd"))
      passed_tests++;
  }

  // Test 5: csr-coo-csc-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_coo, C_csc, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-coo-csc-snd"))
      passed_tests++;
  }

  // Test 6: csr-coo-coo-snd
  {
    reset_tensor(A_csr);
    hadamard_transpose(A_csr, B_coo, C_coo, snd);
    total_tests++;
    if (verify_result_csr(A_csr, "csr-coo-coo-snd"))
      passed_tests++;
  }

  // Test 7: csc-csc-csr-snd
  {
    reset_tensor(A_csc);
    hadamard_transpose(A_csc, B_csc, C_csr, snd);
    total_tests++;
    if (verify_result_csc(A_csc, "csc-csc-csr-snd"))
      passed_tests++;
  }

  // Test 8: csc-csc-csc-snd
  {
    reset_tensor(A_csc);
    hadamard_transpose(A_csc, B_csc, C_csc, snd);
    total_tests++;
    if (verify_result_csc(A_csc, "csc-csc-csc-snd"))
      passed_tests++;
  }

  // Test 9: csc-csc-coo-snd
  {
    reset_tensor(A_csc);
    hadamard_transpose(A_csc, B_csc, C_coo, snd);
    total_tests++;
    if (verify_result_csc(A_csc, "csc-csc-coo-snd"))
      passed_tests++;
  }

  // Free all test data
  free_tensor(A_csr);
  free_tensor(A_csc);
  free_tensor(B_csr);
  free_tensor(C_csr);
  free_tensor(B_csc);
  free_tensor(C_csc);
  free_tensor(B_coo);
  free_tensor(C_coo);

  printf("\n=================================\n");
  printf("Test Results: %d/%d passed\n", passed_tests, total_tests);

  return (passed_tests == total_tests) ? 0 : 1;
}

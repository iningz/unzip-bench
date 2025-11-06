#include "hadamard_transpose.h"

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=CSR, FORMAT_C=CSR
// =============================================================================

#if defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSR)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSR, locate C(j,i) in CSR, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct csr *B, struct csr *C) {
  for (size_t i = 0; i < B->lvl1_size; ++i) {
    size_t b_row_start = B->lvl2_pos[i];
    size_t b_row_end = B->lvl2_pos[i + 1];
    for (size_t b_idx = b_row_start; b_idx < b_row_end; ++b_idx) {
      size_t j = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search row j of C for column i
      size_t c_row_start = C->lvl2_pos[j];
      size_t c_row_end = C->lvl2_pos[j + 1];
      for (size_t c_idx = c_row_start; c_idx < c_row_end; ++c_idx) {
        if (C->lvl2_crd[c_idx] == i) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = j;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#elif defined(SEARCH_B)
#define IMPLEMENTED
// Iterate C(j,i) in CSR, locate B(i,j) in CSR, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct csr *B, struct csr *C) {
  for (size_t j = 0; j < C->lvl1_size; ++j) {
    size_t c_row_start = C->lvl2_pos[j];
    size_t c_row_end = C->lvl2_pos[j + 1];
    for (size_t c_idx = c_row_start; c_idx < c_row_end; ++c_idx) {
      size_t i = C->lvl2_crd[c_idx];
      double c_val = C->vals[c_idx];
      // Locate B(i,j): search row i of B for column j
      size_t b_row_start = B->lvl2_pos[i];
      size_t b_row_end = B->lvl2_pos[i + 1];
      for (size_t b_idx = b_row_start; b_idx < b_row_end; ++b_idx) {
        if (B->lvl2_crd[b_idx] == j) {
          double b_val = B->vals[b_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = j;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[j + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=CSR, FORMAT_C=CSC
// =============================================================================

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSC)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSR, locate C(j,i) in CSC, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct csr *B, struct csc *C) {
  for (size_t i = 0; i < B->lvl1_size; ++i) {
    size_t b_row_start = B->lvl2_pos[i];
    size_t b_row_end = B->lvl2_pos[i + 1];
    for (size_t b_idx = b_row_start; b_idx < b_row_end; ++b_idx) {
      size_t j = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search column i of C for row j
      size_t c_col_start = C->lvl2_pos[i];
      size_t c_col_end = C->lvl2_pos[i + 1];
      for (size_t c_idx = c_col_start; c_idx < c_col_end; ++c_idx) {
        if (C->lvl2_crd[c_idx] == j) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = j;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#elif defined(SEARCH_B)
#define IMPLEMENTED
// Iterate C(j,i) in CSC, locate B(i,j) in CSR, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct csr *B, struct csc *C) {
  for (size_t i = 0; i < C->lvl1_size; ++i) {
    size_t c_col_start = C->lvl2_pos[i];
    size_t c_col_end = C->lvl2_pos[i + 1];
    for (size_t c_idx = c_col_start; c_idx < c_col_end; ++c_idx) {
      size_t j = C->lvl2_crd[c_idx];
      double c_val = C->vals[c_idx];
      // Locate B(i,j): search row i of B for column j
      size_t b_row_start = B->lvl2_pos[i];
      size_t b_row_end = B->lvl2_pos[i + 1];
      for (size_t b_idx = b_row_start; b_idx < b_row_end; ++b_idx) {
        if (B->lvl2_crd[b_idx] == j) {
          double b_val = B->vals[b_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = j;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=CSR, FORMAT_C=COO
// =============================================================================

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_COO)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSR, locate C(j,i) in COO, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct csr *B, struct coo *C) {
  for (size_t i = 0; i < B->lvl1_size; ++i) {
    size_t b_row_start = B->lvl2_pos[i];
    size_t b_row_end = B->lvl2_pos[i + 1];
    for (size_t b_idx = b_row_start; b_idx < b_row_end; ++b_idx) {
      size_t j = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search COO for entry (j,i)
      for (size_t c_idx = 0; c_idx < C->lvl1_nnz; ++c_idx) {
        if (C->lvl1_crd[c_idx] == j && C->lvl2_crd[c_idx] == i) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = j;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=COO, FORMAT_C=CSR
// =============================================================================

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSR)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in COO, locate C(j,i) in CSR, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct coo *B, struct csr *C) {
  for (size_t i = 0; i < A->lvl1_size; ++i) {
    for (size_t b_idx = 0; b_idx < B->lvl1_nnz; ++b_idx) {
      if (B->lvl1_crd[b_idx] == i) {
        size_t j = B->lvl2_crd[b_idx];
        double b_val = B->vals[b_idx];
        // Locate C(j,i): search row j of C for column i
        size_t c_row_start = C->lvl2_pos[j];
        size_t c_row_end = C->lvl2_pos[j + 1];
        for (size_t c_idx = c_row_start; c_idx < c_row_end; ++c_idx) {
          if (C->lvl2_crd[c_idx] == i) {
            double c_val = C->vals[c_idx];
            size_t nnz = A->lvl2_nnz;
            A->lvl2_crd[nnz] = j;
            A->vals[nnz] = b_val * c_val;
            A->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=COO, FORMAT_C=CSC
// =============================================================================

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSC)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in COO, locate C(j,i) in CSC, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct coo *B, struct csc *C) {
  for (size_t i = 0; i < A->lvl1_size; ++i) {
    for (size_t b_idx = 0; b_idx < B->lvl1_nnz; ++b_idx) {
      if (B->lvl1_crd[b_idx] == i) {
        size_t j = B->lvl2_crd[b_idx];
        double b_val = B->vals[b_idx];
        // Locate C(j,i): search column i of C for row j
        size_t c_col_start = C->lvl2_pos[i];
        size_t c_col_end = C->lvl2_pos[i + 1];
        for (size_t c_idx = c_col_start; c_idx < c_col_end; ++c_idx) {
          if (C->lvl2_crd[c_idx] == j) {
            double c_val = C->vals[c_idx];
            size_t nnz = A->lvl2_nnz;
            A->lvl2_crd[nnz] = j;
            A->vals[nnz] = b_val * c_val;
            A->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSR, FORMAT_B=COO, FORMAT_C=COO
// =============================================================================

#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_COO)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in COO, locate C(j,i) in COO, output A(i,j) in CSR
void hadamard_transpose(struct csr *A, struct coo *B, struct coo *C) {
  for (size_t i = 0; i < A->lvl1_size; ++i) {
    for (size_t b_idx = 0; b_idx < B->lvl1_nnz; ++b_idx) {
      if (B->lvl1_crd[b_idx] == i) {
        size_t j = B->lvl2_crd[b_idx];
        double b_val = B->vals[b_idx];
        // Locate C(j,i): search COO for entry (j,i)
        for (size_t c_idx = 0; c_idx < C->lvl1_nnz; ++c_idx) {
          if (C->lvl1_crd[c_idx] == j && C->lvl2_crd[c_idx] == i) {
            double c_val = C->vals[c_idx];
            size_t nnz = A->lvl2_nnz;
            A->lvl2_crd[nnz] = j;
            A->vals[nnz] = b_val * c_val;
            A->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    A->lvl2_pos[i + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSC, FORMAT_B=CSC, FORMAT_C=CSR
// =============================================================================

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSR)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSC, locate C(j,i) in CSR, output A(i,j) in CSC
void hadamard_transpose(struct csc *A, struct csc *B, struct csr *C) {
  for (size_t j = 0; j < B->lvl1_size; ++j) {
    size_t b_col_start = B->lvl2_pos[j];
    size_t b_col_end = B->lvl2_pos[j + 1];
    for (size_t b_idx = b_col_start; b_idx < b_col_end; ++b_idx) {
      size_t i = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search row j of C for column i
      size_t c_row_start = C->lvl2_pos[j];
      size_t c_row_end = C->lvl2_pos[j + 1];
      for (size_t c_idx = c_row_start; c_idx < c_row_end; ++c_idx) {
        if (C->lvl2_crd[c_idx] == i) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = i;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[j + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSC, FORMAT_B=CSC, FORMAT_C=CSC
// =============================================================================

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSC)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSC, locate C(j,i) in CSR, output A(i,j) in CSC
void hadamard_transpose(struct csc *A, struct csc *B, struct csc *C) {
  for (size_t j = 0; j < B->lvl1_size; ++j) {
    size_t b_col_start = B->lvl2_pos[j];
    size_t b_col_end = B->lvl2_pos[j + 1];
    for (size_t b_idx = b_col_start; b_idx < b_col_end; ++b_idx) {
      size_t i = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search column i of C for row j
      size_t c_col_start = C->lvl2_pos[i];
      size_t c_col_end = C->lvl2_pos[i + 1];
      for (size_t c_idx = c_col_start; c_idx < c_col_end; ++c_idx) {
        if (C->lvl2_crd[c_idx] == j) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = i;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[j + 1] = A->lvl2_nnz;
  }
}
#endif

// =============================================================================
// FORMAT_A=CSC, FORMAT_B=CSC, FORMAT_C=COO
// =============================================================================

#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_COO)
#if defined(SEARCH_C)
#define IMPLEMENTED
// Iterate B(i,j) in CSC, locate C(j,i) in COO, output A(i,j) in CSC
void hadamard_transpose(struct csc *A, struct csc *B, struct coo *C) {
  for (size_t j = 0; j < B->lvl1_size; ++j) {
    size_t b_col_start = B->lvl2_pos[j];
    size_t b_col_end = B->lvl2_pos[j + 1];
    for (size_t b_idx = b_col_start; b_idx < b_col_end; ++b_idx) {
      size_t i = B->lvl2_crd[b_idx];
      double b_val = B->vals[b_idx];
      // Locate C(j,i): search COO for entry (j,i)
      for (size_t c_idx = 0; c_idx < C->lvl1_nnz; ++c_idx) {
        if (C->lvl1_crd[c_idx] == j && C->lvl2_crd[c_idx] == i) {
          double c_val = C->vals[c_idx];
          size_t nnz = A->lvl2_nnz;
          A->lvl2_crd[nnz] = i;
          A->vals[nnz] = b_val * c_val;
          A->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    A->lvl2_pos[j + 1] = A->lvl2_nnz;
  }
}
#endif

#endif

#ifndef IMPLEMENTED
#error "Not implemented"
#endif

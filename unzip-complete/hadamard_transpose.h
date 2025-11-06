#ifndef HADAMARD_TRANSPOSE_H
#define HADAMARD_TRANSPOSE_H

#include "tensor_formats.h"

// Compile-time configuration flags:
// FORMAT_A: CSR, CSC, COO
// FORMAT_B: CSR, CSC, COO
// FORMAT_C: CSR, CSC, COO
// SEARCH: B, C (which tensor to iterate first)

// The actual implementation is selected at compile time based on the flags above.
// Only one implementation will be compiled and linked.

// Function signature - actual types depend on compile-time flags
#if defined(FORMAT_A_CSR)
#if defined(FORMAT_B_CSR)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csr *A, struct csr *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csr *A, struct csr *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csr *A, struct csr *B, struct coo *C);
#endif
#elif defined(FORMAT_B_CSC)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csr *A, struct csc *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csr *A, struct csc *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csr *A, struct csc *B, struct coo *C);
#endif
#elif defined(FORMAT_B_COO)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csr *A, struct coo *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csr *A, struct coo *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csr *A, struct coo *B, struct coo *C);
#endif
#endif
#elif defined(FORMAT_A_CSC)
#if defined(FORMAT_B_CSR)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csc *A, struct csr *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csc *A, struct csr *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csc *A, struct csr *B, struct coo *C);
#endif
#elif defined(FORMAT_B_CSC)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csc *A, struct csc *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csc *A, struct csc *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csc *A, struct csc *B, struct coo *C);
#endif
#elif defined(FORMAT_B_COO)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct csc *A, struct coo *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct csc *A, struct coo *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct csc *A, struct coo *B, struct coo *C);
#endif
#endif
#elif defined(FORMAT_A_COO)
#if defined(FORMAT_B_CSR)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct coo *A, struct csr *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct coo *A, struct csr *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct coo *A, struct csr *B, struct coo *C);
#endif
#elif defined(FORMAT_B_CSC)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct coo *A, struct csc *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct coo *A, struct csc *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct coo *A, struct csc *B, struct coo *C);
#endif
#elif defined(FORMAT_B_COO)
#if defined(FORMAT_C_CSR)
void hadamard_transpose(struct coo *A, struct coo *B, struct csr *C);
#elif defined(FORMAT_C_CSC)
void hadamard_transpose(struct coo *A, struct coo *B, struct csc *C);
#elif defined(FORMAT_C_COO)
void hadamard_transpose(struct coo *A, struct coo *B, struct coo *C);
#endif
#endif
#endif

#endif /* HADAMARD_TRANSPOSE_H */

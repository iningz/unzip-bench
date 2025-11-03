#ifndef HADAMARD_TRANSPOSE_H
#define HADAMARD_TRANSPOSE_H

#include "tensor_formats.h"

// Nested _Generic dispatch for hadamard_transpose
#define SELECT_C(A, B, C, A_FMT, B_FMT, SEARCH)                                                                        \
  _Generic((C),                                                                                                        \
      struct csr *: _k_##A_FMT##_##B_FMT##_csr_##SEARCH,                                                               \
      struct csc *: _k_##A_FMT##_##B_FMT##_csc_##SEARCH,                                                               \
      struct coo *: _k_##A_FMT##_##B_FMT##_coo_##SEARCH)

#define SELECT_B(A, B, C, A_FMT, SEARCH)                                                                               \
  _Generic((B),                                                                                                        \
      struct csr *: SELECT_C(A, B, C, A_FMT, csr, SEARCH),                                                             \
      struct csc *: SELECT_C(A, B, C, A_FMT, csc, SEARCH),                                                             \
      struct coo *: SELECT_C(A, B, C, A_FMT, coo, SEARCH))

#define hadamard_transpose(A, B, C, SEARCH)                                                                            \
  _Generic((A),                                                                                                        \
      struct csr *: SELECT_B(A, B, C, csr, SEARCH),                                                                    \
      struct csc *: SELECT_B(A, B, C, csc, SEARCH),                                                                    \
      struct coo *: SELECT_B(A, B, C, coo, SEARCH))(A, B, C)

// A=CSR, B=CSR, C=*
void _k_csr_csr_csr_fst(struct csr *res, struct csr *t1, struct csr *t2);
void _k_csr_csr_csr_snd(struct csr *res, struct csr *t1, struct csr *t2);
void _k_csr_csr_csc_fst(struct csr *res, struct csr *t1, struct csc *t2);
void _k_csr_csr_csc_snd(struct csr *res, struct csr *t1, struct csc *t2);
void _k_csr_csr_coo_fst(struct csr *res, struct csr *t1, struct coo *t2);
void _k_csr_csr_coo_snd(struct csr *res, struct csr *t1, struct coo *t2);

// A=CSR, B=CSC, C=*
void _k_csr_csc_csr_fst(struct csr *res, struct csc *t1, struct csr *t2);
void _k_csr_csc_csr_snd(struct csr *res, struct csc *t1, struct csr *t2);
void _k_csr_csc_csc_fst(struct csr *res, struct csc *t1, struct csc *t2);
void _k_csr_csc_csc_snd(struct csr *res, struct csc *t1, struct csc *t2);
void _k_csr_csc_coo_fst(struct csr *res, struct csc *t1, struct coo *t2);
void _k_csr_csc_coo_snd(struct csr *res, struct csc *t1, struct coo *t2);

// A=CSR, B=COO, C=*
void _k_csr_coo_csr_fst(struct csr *res, struct coo *t1, struct csr *t2);
void _k_csr_coo_csr_snd(struct csr *res, struct coo *t1, struct csr *t2);
void _k_csr_coo_csc_fst(struct csr *res, struct coo *t1, struct csc *t2);
void _k_csr_coo_csc_snd(struct csr *res, struct coo *t1, struct csc *t2);
void _k_csr_coo_coo_fst(struct csr *res, struct coo *t1, struct coo *t2);
void _k_csr_coo_coo_snd(struct csr *res, struct coo *t1, struct coo *t2);

// A=CSC, B=CSR, C=*
void _k_csc_csr_csr_fst(struct csc *res, struct csr *t1, struct csr *t2);
void _k_csc_csr_csr_snd(struct csc *res, struct csr *t1, struct csr *t2);
void _k_csc_csr_csc_fst(struct csc *res, struct csr *t1, struct csc *t2);
void _k_csc_csr_csc_snd(struct csc *res, struct csr *t1, struct csc *t2);
void _k_csc_csr_coo_fst(struct csc *res, struct csr *t1, struct coo *t2);
void _k_csc_csr_coo_snd(struct csc *res, struct csr *t1, struct coo *t2);

// A=CSC, B=CSC, C=*
void _k_csc_csc_csr_fst(struct csc *res, struct csc *t1, struct csr *t2);
void _k_csc_csc_csr_snd(struct csc *res, struct csc *t1, struct csr *t2);
void _k_csc_csc_csc_fst(struct csc *res, struct csc *t1, struct csc *t2);
void _k_csc_csc_csc_snd(struct csc *res, struct csc *t1, struct csc *t2);
void _k_csc_csc_coo_fst(struct csc *res, struct csc *t1, struct coo *t2);
void _k_csc_csc_coo_snd(struct csc *res, struct csc *t1, struct coo *t2);

// A=CSC, B=COO, C=*
void _k_csc_coo_csr_fst(struct csc *res, struct coo *t1, struct csr *t2);
void _k_csc_coo_csr_snd(struct csc *res, struct coo *t1, struct csr *t2);
void _k_csc_coo_csc_fst(struct csc *res, struct coo *t1, struct csc *t2);
void _k_csc_coo_csc_snd(struct csc *res, struct coo *t1, struct csc *t2);
void _k_csc_coo_coo_fst(struct csc *res, struct coo *t1, struct coo *t2);
void _k_csc_coo_coo_snd(struct csc *res, struct coo *t1, struct coo *t2);

// A=COO, B=CSR, C=*
void _k_coo_csr_csr_fst(struct coo *res, struct csr *t1, struct csr *t2);
void _k_coo_csr_csr_snd(struct coo *res, struct csr *t1, struct csr *t2);
void _k_coo_csr_csc_fst(struct coo *res, struct csr *t1, struct csc *t2);
void _k_coo_csr_csc_snd(struct coo *res, struct csr *t1, struct csc *t2);
void _k_coo_csr_coo_fst(struct coo *res, struct csr *t1, struct coo *t2);
void _k_coo_csr_coo_snd(struct coo *res, struct csr *t1, struct coo *t2);

// A=COO, B=CSC, C=*
void _k_coo_csc_csr_fst(struct coo *res, struct csc *t1, struct csr *t2);
void _k_coo_csc_csr_snd(struct coo *res, struct csc *t1, struct csr *t2);
void _k_coo_csc_csc_fst(struct coo *res, struct csc *t1, struct csc *t2);
void _k_coo_csc_csc_snd(struct coo *res, struct csc *t1, struct csc *t2);
void _k_coo_csc_coo_fst(struct coo *res, struct csc *t1, struct coo *t2);
void _k_coo_csc_coo_snd(struct coo *res, struct csc *t1, struct coo *t2);

// A=COO, B=COO, C=*
void _k_coo_coo_csr_fst(struct coo *res, struct coo *t1, struct csr *t2);
void _k_coo_coo_csr_snd(struct coo *res, struct coo *t1, struct csr *t2);
void _k_coo_coo_csc_fst(struct coo *res, struct coo *t1, struct csc *t2);
void _k_coo_coo_csc_snd(struct coo *res, struct coo *t1, struct csc *t2);
void _k_coo_coo_coo_fst(struct coo *res, struct coo *t1, struct coo *t2);
void _k_coo_coo_coo_snd(struct coo *res, struct coo *t1, struct coo *t2);

#endif /* HADAMARD_TRANSPOSE_H */

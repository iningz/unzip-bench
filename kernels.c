#include "formats.h"
#include <stdio.h>
#include <stdlib.h>

/* Kernel implementation: A(i,j) = B(i,j) * C(j,i) */
void hadamard_transpose(struct csr *t1, struct csr *t2, struct csr *res) {
  for (size_t t1_lvl1_idx = 0; t1_lvl1_idx < t1->lvl1_size; ++t1_lvl1_idx) {
    // Iterate over i in B(i,j)
    size_t t1_lvl1_pos_start = t1->lvl1_pos[t1_lvl1_idx];
    size_t t1_lvl1_pos_end = t1->lvl1_pos[t1_lvl1_idx + 1];
    for (size_t t1_lvl1_pos_idx = t1_lvl1_pos_start; t1_lvl1_pos_idx < t1_lvl1_pos_end; ++t1_lvl1_pos_idx) {
      size_t t1_lvl2_crd = t1->lvl2_crd[t1_lvl1_pos_idx];
      double t1_val = t1->vals[t1_lvl1_pos_idx];
      // Locate matching j in C(j,i)
      size_t t2_lvl1_pos_start = t2->lvl1_pos[t1_lvl2_crd];
      size_t t2_lvl1_pos_end = t2->lvl1_pos[t1_lvl2_crd + 1];
      for (size_t t2_lvl1_pos_idx = t2_lvl1_pos_start; t2_lvl1_pos_idx < t2_lvl1_pos_end; ++t2_lvl1_pos_idx) {
        size_t t2_lvl2_crd = t2->lvl2_crd[t2_lvl1_pos_idx];
        if (t2_lvl2_crd == t1_lvl1_idx) {
          double t2_val = t2->vals[t2_lvl1_pos_idx];
          // Set A(i,j)
          if (t1_val != 0.0 && t2_val != 0.0) {
            size_t nnz = res->lvl2_nnz;
            res->lvl2_crd[nnz] = t1_lvl2_crd;
            res->vals[nnz] = t1_val * t2_val;
            res->lvl2_nnz = nnz + 1;
          }
        }
      }
    }
    res->lvl1_pos[t1_lvl1_idx + 1] = res->lvl2_nnz;
  }
}

/* A(i, j) = B(i, k) * C(k, j) */
void matmul(struct csr *t1, struct csr *t2, struct csr *res) {
  // Allocate dense accumulation buffers for output dimension j (columns)
  // Reducing over dimension k (shared dimension between B and C)
  double *lvl2_acc = (double *)calloc(res->lvl2_size, sizeof(double));
  size_t *lvl2_mkr = (size_t *)calloc(res->lvl2_size, sizeof(size_t));

  for (size_t t1_lvl1_idx = 0; t1_lvl1_idx < t1->lvl1_size; ++t1_lvl1_idx) {
    // Phase 1: Accumulate into dense buffer
    // Iterate over i in B(i,k)
    size_t t1_lvl1_pos_start = t1->lvl1_pos[t1_lvl1_idx];
    size_t t1_lvl1_pos_end = t1->lvl1_pos[t1_lvl1_idx + 1];
    for (size_t t1_lvl1_pos_idx = t1_lvl1_pos_start; t1_lvl1_pos_idx < t1_lvl1_pos_end; ++t1_lvl1_pos_idx) {
      size_t t1_lvl2_crd = t1->lvl2_crd[t1_lvl1_pos_idx];
      double t1_val = t1->vals[t1_lvl1_pos_idx];
      // Iterate over k in C(k,j)
      size_t t2_lvl1_pos_start = t2->lvl1_pos[t1_lvl2_crd];
      size_t t2_lvl1_pos_end = t2->lvl1_pos[t1_lvl2_crd + 1];
      for (size_t t2_lvl1_pos_idx = t2_lvl1_pos_start; t2_lvl1_pos_idx < t2_lvl1_pos_end; ++t2_lvl1_pos_idx) {
        size_t t2_lvl2_crd = t2->lvl2_crd[t2_lvl1_pos_idx];
        double t2_val = t2->vals[t2_lvl1_pos_idx];
        // Accumulate into buffer for A(i,j)
        lvl2_mkr[t2_lvl2_crd] = t1_lvl1_idx + 1; // Mark as touched in this dim
        lvl2_acc[t2_lvl2_crd] += t1_val * t2_val;
      }
    }

    // Phase 2: Compress buffer into CSR output
    for (size_t lvl2_idx = 0; lvl2_idx < res->lvl2_size; ++lvl2_idx) {
      size_t lvl2_marker_crd = lvl2_mkr[lvl2_idx];
      double t2_val = lvl2_acc[lvl2_idx];
      if (lvl2_marker_crd == t1_lvl1_idx + 1 && t2_val != 0.0) {
        size_t nnz = res->lvl2_nnz;
        res->lvl2_crd[nnz] = lvl2_idx;
        res->vals[nnz] = t2_val;
        res->lvl2_nnz = nnz + 1;
        lvl2_acc[lvl2_idx] = 0.0; // Reset for next row
      }
    }
    res->lvl1_pos[t1_lvl1_idx + 1] = res->lvl2_nnz;
  }

  free(lvl2_acc);
  free(lvl2_mkr);
}

/* A(i, j) = B(i, k) * C(k, j) * D(k, j) */
void matmul_hadamard(struct csr *t1, struct csr *t2, struct csr *t3, struct csr *res) {
  // Allocate dense accumulation buffer for one row
  double *lvl2_acc = (double *)calloc(res->lvl2_size, sizeof(double));
  size_t *lvl2_mkr = (size_t *)calloc(res->lvl2_size, sizeof(size_t));

  for (size_t t1_lvl1_idx = 0; t1_lvl1_idx < t1->lvl1_size; ++t1_lvl1_idx) {
    // Phase 1: Accumulate into dense buffer
    // Iterate over i in B(i,k)
    size_t t1_lvl1_pos_start = t1->lvl1_pos[t1_lvl1_idx];
    size_t t1_lvl1_pos_end = t1->lvl1_pos[t1_lvl1_idx + 1];
    for (size_t t1_lvl1_pos_idx = t1_lvl1_pos_start; t1_lvl1_pos_idx < t1_lvl1_pos_end; ++t1_lvl1_pos_idx) {
      size_t t1_lvl2_crd = t1->lvl2_crd[t1_lvl1_pos_idx];
      double t1_val = t1->vals[t1_lvl1_pos_idx];
      // Iterate over k in C(k,j)
      size_t t2_lvl1_pos_start = t2->lvl1_pos[t1_lvl2_crd];
      size_t t2_lvl1_pos_end = t2->lvl1_pos[t1_lvl2_crd + 1];
      for (size_t t2_lvl1_pos_idx = t2_lvl1_pos_start; t2_lvl1_pos_idx < t2_lvl1_pos_end; ++t2_lvl1_pos_idx) {
        size_t t2_lvl2_crd = t2->lvl2_crd[t2_lvl1_pos_idx];
        double t2_val = t2->vals[t2_lvl1_pos_idx];
        // Locate matching j in D(k,j)
        size_t t3_lvl1_pos_start = t3->lvl1_pos[t1_lvl2_crd];
        size_t t3_lvl1_pos_end = t3->lvl1_pos[t1_lvl2_crd + 1];
        for (size_t t3_lvl1_pos_idx = t3_lvl1_pos_start; t3_lvl1_pos_idx < t3_lvl1_pos_end; ++t3_lvl1_pos_idx) {
          size_t t3_lvl2_crd = t3->lvl2_crd[t3_lvl1_pos_idx];
          if (t3_lvl2_crd == t2_lvl2_crd) {
            double t3_val = t3->vals[t3_lvl1_pos_idx];
            // Accumulate into buffer for A(i,j)
            lvl2_mkr[t2_lvl2_crd] = t1_lvl1_idx + 1;
            lvl2_acc[t2_lvl2_crd] += t1_val * t2_val * t3_val;
            break;
          }
        }
      }
    }

    // Phase 2: Compress buffer into CSR output
    for (size_t t2_lvl1_idx = 0; t2_lvl1_idx < t2->lvl1_size; ++t2_lvl1_idx) {
      size_t lvl2_marker_crd = lvl2_mkr[t2_lvl1_idx];
      double t2_val = lvl2_acc[t2_lvl1_idx];
      if (lvl2_marker_crd == t1_lvl1_idx + 1 && t2_val != 0.0) {
        size_t nnz = res->lvl2_nnz;
        res->lvl2_crd[nnz] = t2_lvl1_idx;
        res->vals[nnz] = t2_val;
        res->lvl2_nnz = nnz + 1;
        lvl2_acc[t2_lvl1_idx] = 0.0; // Reset for next row
      }
    }

    res->lvl1_pos[t1_lvl1_idx + 1] = res->lvl2_nnz;
  }

  free(lvl2_acc);
  free(lvl2_mkr);
}

/* y(i) = B(i, j) * C(j, i) */
void hadamard_transpose_reduce(struct csr *t1, struct csr *t2, struct dense *res) {
  for (size_t t1_lvl1_idx = 0; t1_lvl1_idx < t1->lvl1_size; ++t1_lvl1_idx) {
    // Iterate over i in B(i,j)
    size_t t1_lvl1_pos_start = t1->lvl1_pos[t1_lvl1_idx];
    size_t t1_lvl1_pos_end = t1->lvl1_pos[t1_lvl1_idx + 1];
    for (size_t t1_lvl1_pos_idx = t1_lvl1_pos_start; t1_lvl1_pos_idx < t1_lvl1_pos_end; ++t1_lvl1_pos_idx) {
      size_t t1_lvl2_crd = t1->lvl2_crd[t1_lvl1_pos_idx];
      double t1_val = t1->vals[t1_lvl1_pos_idx];
      // Locate matching j in C(j,i)
      size_t t2_lvl1_pos_start = t2->lvl1_pos[t1_lvl2_crd];
      size_t t2_lvl1_pos_end = t2->lvl1_pos[t1_lvl2_crd + 1];
      for (size_t t2_lvl1_pos_idx = t2_lvl1_pos_start; t2_lvl1_pos_idx < t2_lvl1_pos_end; ++t2_lvl1_pos_idx) {
        size_t t2_lvl2_crd = t2->lvl2_crd[t2_lvl1_pos_idx];
        if (t2_lvl2_crd == t1_lvl1_idx) {
          double t2_val = t2->vals[t2_lvl1_pos_idx];
          // Accumulate into y(i)
          res->vals[t1_lvl1_idx] += t1_val * t2_val;
          break;
        }
      }
    }
  }
}

// y(i) = B(i, j, k) * C(i, k, j)
void permute_contract(struct csf *t1, struct csf *t2, struct dense *res) {
  // Iterate over i in B and C
  for (size_t t1_lvl1_idx = 0; t1_lvl1_idx < t1->lvl1_size; ++t1_lvl1_idx) {
    size_t t2_lvl1_pos_start = t2->lvl1_pos[t1_lvl1_idx];
    size_t t2_lvl1_pos_end = t2->lvl1_pos[t1_lvl1_idx + 1];
    // Iterate over j in B(i,j,k)
    for (size_t t1_lvl1_pos_idx = t1->lvl1_pos[t1_lvl1_idx]; t1_lvl1_pos_idx < t1->lvl1_pos[t1_lvl1_idx + 1]; ++t1_lvl1_pos_idx) {
      size_t t1_lvl2_crd = t1->lvl2_crd[t1_lvl1_pos_idx];
      size_t t1_lvl2_pos_start = t1->lvl2_pos[t1_lvl1_pos_idx];
      size_t t1_lvl2_pos_end = t1->lvl2_pos[t1_lvl1_pos_idx + 1];
      // Iterate over k in C(i,k,j)
      for (size_t t2_lvl1_pos_idx = t2->lvl1_pos[t1_lvl1_idx]; t2_lvl1_pos_idx < t2->lvl1_pos[t1_lvl1_idx + 1]; ++t2_lvl1_pos_idx) {
        size_t t2_lvl2_crd = t2->lvl2_crd[t2_lvl1_pos_idx]; // k dimension in C(i,k,j)
        size_t t2_lvl2_pos_start = t2->lvl2_pos[t2_lvl1_pos_idx];
        size_t t2_lvl2_pos_end = t2->lvl2_pos[t2_lvl1_pos_idx + 1];
        // Locate matching k in B(i,j,k)
        for (size_t t1_lvl3_crd_idx = t1_lvl2_pos_start; t1_lvl3_crd_idx < t1_lvl2_pos_end; ++t1_lvl3_crd_idx) {
          size_t t1_lvl3_crd = t1->lvl3_crd[t1_lvl3_crd_idx];
          if (t1_lvl3_crd == t2_lvl2_crd) { // k indices match
            double t1_val = t1->vals[t1_lvl3_crd_idx];
            // Locate matching j in C(i,k,j)
            for (size_t t2_lvl3_crd_idx = t2_lvl2_pos_start; t2_lvl3_crd_idx < t2_lvl2_pos_end; ++t2_lvl3_crd_idx) {
              size_t t2_lvl3_crd = t2->lvl3_crd[t2_lvl3_crd_idx];
              if (t2_lvl3_crd == t1_lvl2_crd) { // j indices match
                double t2_val = t2->vals[t2_lvl3_crd_idx];
                // Accumulate into y(i)
                res->vals[t1_lvl1_idx] += t1_val * t2_val;
                break;
              }
            }
            break;
          }
        }
      }
    }
  }
}

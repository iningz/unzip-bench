// A(i,j) = B(i,j) * C(j,i)

#pragma GCC diagnostic ignored "-Wunused-parameter"

#include "hadamard_transpose.h"
#include "tensor_formats.h"
#include <stdlib.h>

// Iterate C(j,i) in CSR, locate B(i,j) in CSR, output A(i,j) in CSR
void _k_csr_csr_csr_fst(struct csr *res, struct csr *t1, struct csr *t2) {
  // Iterate C: for each row j
  for (size_t j = 0; j < t2->lvl1_size; ++j) {
    size_t t2_row_start = t2->lvl2_pos[j];
    size_t t2_row_end = t2->lvl2_pos[j + 1];
    // Iterate C: for each column i in row j, getting C(j,i)
    for (size_t t2_idx = t2_row_start; t2_idx < t2_row_end; ++t2_idx) {
      size_t i = t2->lvl2_crd[t2_idx];
      double c_val = t2->vals[t2_idx];
      // Locate B(i,j): search row i of B for column j
      size_t t1_row_start = t1->lvl2_pos[i];
      size_t t1_row_end = t1->lvl2_pos[i + 1];
      for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
        if (t1->lvl2_crd[t1_idx] == j) {
          double b_val = t1->vals[t1_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
        }
      }
    }
    res->lvl2_pos[j + 1] = res->lvl2_nnz;
  }
}

// Iterate B(i,j) in CSR, locate C(j,i) in CSR, output A(i,j) in CSR
void _k_csr_csr_csr_snd(struct csr *res, struct csr *t1, struct csr *t2) {
  // Iterate B: for each row i
  for (size_t i = 0; i < t1->lvl1_size; ++i) {
    size_t t1_row_start = t1->lvl2_pos[i];
    size_t t1_row_end = t1->lvl2_pos[i + 1];
    // Iterate B: for each column j in row i, getting B(i,j)
    for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
      size_t j = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): search row j of C for column i
      size_t t2_row_start = t2->lvl2_pos[j];
      size_t t2_row_end = t2->lvl2_pos[j + 1];
      for (size_t t2_idx = t2_row_start; t2_idx < t2_row_end; ++t2_idx) {
        if (t2->lvl2_crd[t2_idx] == i) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

// Iterate C(j,i) in CSC, locate B(i,j) in CSR, output A(i,j) in CSR
void _k_csr_csr_csc_fst(struct csr *res, struct csr *t1, struct csc *t2) {
  // Iterate C: for each column i (CSC stores by columns)
  for (size_t i = 0; i < t2->lvl1_size; ++i) {
    size_t t2_col_start = t2->lvl2_pos[i];
    size_t t2_col_end = t2->lvl2_pos[i + 1];
    // Iterate C: for each row j in column i, getting C(j,i)
    for (size_t t2_idx = t2_col_start; t2_idx < t2_col_end; ++t2_idx) {
      size_t j = t2->lvl2_crd[t2_idx];
      double c_val = t2->vals[t2_idx];
      // Locate B(i,j): search row i of B for column j
      size_t t1_row_start = t1->lvl2_pos[i];
      size_t t1_row_end = t1->lvl2_pos[i + 1];
      for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
        if (t1->lvl2_crd[t1_idx] == j) {
          double b_val = t1->vals[t1_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

// Iterate B(i,j) in CSR, locate C(j,i) in CSC, output A(i,j) in CSR
void _k_csr_csr_csc_snd(struct csr *res, struct csr *t1, struct csc *t2) {
  // Iterate B: for each row i
  for (size_t i = 0; i < t1->lvl1_size; ++i) {
    size_t t1_row_start = t1->lvl2_pos[i];
    size_t t1_row_end = t1->lvl2_pos[i + 1];
    // Iterate B: for each column j in row i, getting B(i,j)
    for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
      size_t j = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): search column i of C for row j (CSC stores by columns)
      size_t t2_col_start = t2->lvl2_pos[i];
      size_t t2_col_end = t2->lvl2_pos[i + 1];
      for (size_t t2_idx = t2_col_start; t2_idx < t2_col_end; ++t2_idx) {
        if (t2->lvl2_crd[t2_idx] == j) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

// Iterate C(j,i) in COO, locate B(i,j) in CSR, output A(i,j) in CSR
void _k_csr_csr_coo_fst(struct csr *res, struct csr *t1, struct coo *t2) {
  // Iterate B: for each row i (output is CSR, must iterate by rows)
  for (size_t i = 0; i < t1->lvl1_size; ++i) {
    size_t t1_row_start = t1->lvl2_pos[i];
    size_t t1_row_end = t1->lvl2_pos[i + 1];
    // Iterate B: for each column j in row i, getting B(i,j)
    for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
      size_t j = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): linear search in COO for (j,i)
      for (size_t t2_idx = 0; t2_idx < t2->lvl1_nnz; ++t2_idx) {
        if (t2->lvl1_crd[t2_idx] == j && t2->lvl2_crd[t2_idx] == i) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

// Iterate B(i,j) in CSR, locate C(j,i) in COO, output A(i,j) in CSR
void _k_csr_csr_coo_snd(struct csr *res, struct csr *t1, struct coo *t2) {
  // Iterate B: for each row i
  for (size_t i = 0; i < t1->lvl1_size; ++i) {
    size_t t1_row_start = t1->lvl2_pos[i];
    size_t t1_row_end = t1->lvl2_pos[i + 1];
    // Iterate B: for each column j in row i, getting B(i,j)
    for (size_t t1_idx = t1_row_start; t1_idx < t1_row_end; ++t1_idx) {
      size_t j = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): linear search in COO for (j,i)
      for (size_t t2_idx = 0; t2_idx < t2->lvl1_nnz; ++t2_idx) {
        if (t2->lvl1_crd[t2_idx] == j && t2->lvl2_crd[t2_idx] == i) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = j;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

void _k_csr_csc_csr_fst(struct csr *res, struct csc *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_csc_csr_snd(struct csr *res, struct csc *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_csc_csc_fst(struct csr *res, struct csc *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_csc_csc_snd(struct csr *res, struct csc *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_csc_coo_fst(struct csr *res, struct csc *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_csc_coo_snd(struct csr *res, struct csc *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_csr_coo_csr_fst(struct csr *res, struct coo *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

// Iterate B(i,j) in COO, locate C(j,i) in CSR, output A(i,j) in CSR
void _k_csr_coo_csr_snd(struct csr *res, struct coo *t1, struct csr *t2) {
  // Output is CSR: must build row-by-row
  for (size_t i = 0; i < res->lvl1_size; ++i) {
    // Iterate B: scan COO for entries with row i, getting B(i,j)
    for (size_t t1_idx = 0; t1_idx < t1->lvl1_nnz; ++t1_idx) {
      if (t1->lvl1_crd[t1_idx] == i) {
        size_t j = t1->lvl2_crd[t1_idx];
        double b_val = t1->vals[t1_idx];
        // Locate C(j,i): search row j of C for column i
        size_t t2_row_start = t2->lvl2_pos[j];
        size_t t2_row_end = t2->lvl2_pos[j + 1];
        for (size_t t2_idx = t2_row_start; t2_idx < t2_row_end; ++t2_idx) {
          if (t2->lvl2_crd[t2_idx] == i) {
            double c_val = t2->vals[t2_idx];
            // Output A(i,j) = B(i,j) * C(j,i)
            size_t nnz = res->lvl2_nnz;
            res->lvl2_crd[nnz] = j;
            res->vals[nnz] = b_val * c_val;
            res->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

void _k_csr_coo_csc_fst(struct csr *res, struct coo *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

// Iterate B(i,j) in COO, locate C(j,i) in CSC, output A(i,j) in CSR
void _k_csr_coo_csc_snd(struct csr *res, struct coo *t1, struct csc *t2) {
  // Output is CSR: must build row-by-row
  for (size_t i = 0; i < res->lvl1_size; ++i) {
    // Iterate B: scan COO for entries with row i, getting B(i,j)
    for (size_t t1_idx = 0; t1_idx < t1->lvl1_nnz; ++t1_idx) {
      if (t1->lvl1_crd[t1_idx] == i) {
        size_t j = t1->lvl2_crd[t1_idx];
        double b_val = t1->vals[t1_idx];
        // Locate C(j,i): search column i of C for row j (CSC stores by columns)
        size_t t2_col_start = t2->lvl2_pos[i];
        size_t t2_col_end = t2->lvl2_pos[i + 1];
        for (size_t t2_idx = t2_col_start; t2_idx < t2_col_end; ++t2_idx) {
          if (t2->lvl2_crd[t2_idx] == j) {
            double c_val = t2->vals[t2_idx];
            // Output A(i,j) = B(i,j) * C(j,i)
            size_t nnz = res->lvl2_nnz;
            res->lvl2_crd[nnz] = j;
            res->vals[nnz] = b_val * c_val;
            res->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

void _k_csr_coo_coo_fst(struct csr *res, struct coo *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

// Iterate B(i,j) in COO, locate C(j,i) in COO, output A(i,j) in CSR
void _k_csr_coo_coo_snd(struct csr *res, struct coo *t1, struct coo *t2) {
  // Output is CSR: must build row-by-row
  for (size_t i = 0; i < res->lvl1_size; ++i) {
    // Iterate B: scan COO for entries with row i, getting B(i,j)
    for (size_t t1_idx = 0; t1_idx < t1->lvl1_nnz; ++t1_idx) {
      if (t1->lvl1_crd[t1_idx] == i) {
        size_t j = t1->lvl2_crd[t1_idx];
        double b_val = t1->vals[t1_idx];
        // Locate C(j,i): linear search in COO for (j,i)
        for (size_t t2_idx = 0; t2_idx < t2->lvl1_nnz; ++t2_idx) {
          if (t2->lvl1_crd[t2_idx] == j && t2->lvl2_crd[t2_idx] == i) {
            double c_val = t2->vals[t2_idx];
            // Output A(i,j) = B(i,j) * C(j,i)
            size_t nnz = res->lvl2_nnz;
            res->lvl2_crd[nnz] = j;
            res->vals[nnz] = b_val * c_val;
            res->lvl2_nnz = nnz + 1;
            break;
          }
        }
      }
    }
    res->lvl2_pos[i + 1] = res->lvl2_nnz;
  }
}

void _k_csc_csr_csr_fst(struct csc *res, struct csr *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csr_csr_snd(struct csc *res, struct csr *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csr_csc_fst(struct csc *res, struct csr *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csr_csc_snd(struct csc *res, struct csr *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csr_coo_fst(struct csc *res, struct csr *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csr_coo_snd(struct csc *res, struct csr *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_csc_csr_fst(struct csc *res, struct csc *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

// csc-csc-csr-snd: Iterate B(i,j) in CSC, locate C(j,i) in CSR, output A(i,j) in CSC
void _k_csc_csc_csr_snd(struct csc *res, struct csc *t1, struct csr *t2) {
  // Output is CSC: must build column-by-column
  // Iterate B: for each column j (CSC stores by columns)
  for (size_t j = 0; j < t1->lvl1_size; ++j) {
    size_t t1_col_start = t1->lvl2_pos[j];
    size_t t1_col_end = t1->lvl2_pos[j + 1];
    // Iterate B: for each row i in column j, getting B(i,j)
    for (size_t t1_idx = t1_col_start; t1_idx < t1_col_end; ++t1_idx) {
      size_t i = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): search row j of C for column i
      size_t t2_row_start = t2->lvl2_pos[j];
      size_t t2_row_end = t2->lvl2_pos[j + 1];
      for (size_t t2_idx = t2_row_start; t2_idx < t2_row_end; ++t2_idx) {
        if (t2->lvl2_crd[t2_idx] == i) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i) in CSC format (column j, row i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = i;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    res->lvl2_pos[j + 1] = res->lvl2_nnz;
  }
}

void _k_csc_csc_csc_fst(struct csc *res, struct csc *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

// Iterate B(i,j) in CSC, locate C(j,i) in CSC, output A(i,j) in CSC
void _k_csc_csc_csc_snd(struct csc *res, struct csc *t1, struct csc *t2) {
  // Output is CSC: must build column-by-column
  // Iterate B: for each column j (CSC stores by columns)
  for (size_t j = 0; j < t1->lvl1_size; ++j) {
    size_t t1_col_start = t1->lvl2_pos[j];
    size_t t1_col_end = t1->lvl2_pos[j + 1];
    // Iterate B: for each row i in column j, getting B(i,j)
    for (size_t t1_idx = t1_col_start; t1_idx < t1_col_end; ++t1_idx) {
      size_t i = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): search column i of C for row j (CSC stores by columns)
      size_t t2_col_start = t2->lvl2_pos[i];
      size_t t2_col_end = t2->lvl2_pos[i + 1];
      for (size_t t2_idx = t2_col_start; t2_idx < t2_col_end; ++t2_idx) {
        if (t2->lvl2_crd[t2_idx] == j) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i) in CSC format (column j, row i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = i;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    res->lvl2_pos[j + 1] = res->lvl2_nnz;
  }
}

void _k_csc_csc_coo_fst(struct csc *res, struct csc *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

// Iterate B(i,j) in CSC, locate C(j,i) in COO, output A(i,j) in CSC
void _k_csc_csc_coo_snd(struct csc *res, struct csc *t1, struct coo *t2) {
  // Output is CSC: must build column-by-column
  // Iterate B: for each column j (CSC stores by columns)
  for (size_t j = 0; j < t1->lvl1_size; ++j) {
    size_t t1_col_start = t1->lvl2_pos[j];
    size_t t1_col_end = t1->lvl2_pos[j + 1];
    // Iterate B: for each row i in column j, getting B(i,j)
    for (size_t t1_idx = t1_col_start; t1_idx < t1_col_end; ++t1_idx) {
      size_t i = t1->lvl2_crd[t1_idx];
      double b_val = t1->vals[t1_idx];
      // Locate C(j,i): linear search in COO for (j,i)
      for (size_t t2_idx = 0; t2_idx < t2->lvl1_nnz; ++t2_idx) {
        if (t2->lvl1_crd[t2_idx] == j && t2->lvl2_crd[t2_idx] == i) {
          double c_val = t2->vals[t2_idx];
          // Output A(i,j) = B(i,j) * C(j,i) in CSC format (column j, row i)
          size_t nnz = res->lvl2_nnz;
          res->lvl2_crd[nnz] = i;
          res->vals[nnz] = b_val * c_val;
          res->lvl2_nnz = nnz + 1;
          break;
        }
      }
    }
    res->lvl2_pos[j + 1] = res->lvl2_nnz;
  }
}

void _k_csc_coo_csr_fst(struct csc *res, struct coo *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_coo_csr_snd(struct csc *res, struct coo *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_coo_csc_fst(struct csc *res, struct coo *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_coo_csc_snd(struct csc *res, struct coo *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_coo_coo_fst(struct csc *res, struct coo *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_csc_coo_coo_snd(struct csc *res, struct coo *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_csr_fst(struct coo *res, struct csr *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_csr_snd(struct coo *res, struct csr *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_csc_fst(struct coo *res, struct csr *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_csc_snd(struct coo *res, struct csr *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_coo_fst(struct coo *res, struct csr *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csr_coo_snd(struct coo *res, struct csr *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_csr_fst(struct coo *res, struct csc *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_csr_snd(struct coo *res, struct csc *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_csc_fst(struct coo *res, struct csc *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_csc_snd(struct coo *res, struct csc *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_coo_fst(struct coo *res, struct csc *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_csc_coo_snd(struct coo *res, struct csc *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_csr_fst(struct coo *res, struct coo *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_csr_snd(struct coo *res, struct coo *t1, struct csr *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_csc_fst(struct coo *res, struct coo *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_csc_snd(struct coo *res, struct coo *t1, struct csc *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_coo_fst(struct coo *res, struct coo *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

void _k_coo_coo_coo_snd(struct coo *res, struct coo *t1, struct coo *t2) {
  // NOT IMPLEMENTED
}

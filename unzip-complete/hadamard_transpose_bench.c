#include "hadamard_transpose.h"
#include "tensor_formats.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

// Configuration
const unsigned int SEED = 42;
#ifdef DEBUG
const size_t MIN_SIZE = 10;
const size_t MAX_SIZE = 100;
const int NUM_SAMPLES = 5;
const int NUM_WARMUP = 0;
const int NUM_RUNS = 1;
#else
const size_t MIN_SIZE = 100;
const size_t MAX_SIZE = 10000;
const int NUM_SAMPLES = 20;
const int NUM_WARMUP = 0;
const int NUM_RUNS = 1;
#endif

const double SPARSITIES[] = {0.05, 0.1, 0.25, 0.5, 0.75};
const size_t NUM_SPARSITIES = sizeof(SPARSITIES) / sizeof(SPARSITIES[0]);

// Generate logarithmically-spaced sizes
static void generate_sizes(size_t *sizes, size_t *count) {
  double log_min = log10(MIN_SIZE);
  double log_max = log10(MAX_SIZE);
  size_t idx = 0;

  for (int s = 0; s < NUM_SAMPLES; ++s) {
    double log_val = log_min + (log_max - log_min) * s / (NUM_SAMPLES - 1);
    size_t size = (size_t)round(pow(10.0, log_val));

    bool duplicated = false;
    for (size_t i = 0; i < idx; ++i) {
      if (sizes[i] == size) {
        duplicated = true;
        break;
      }
    }

    if (!duplicated) {
      sizes[idx++] = size;
    }
  }
  *count = idx;
}

// Get use CPU time in microseconds using getrusage
static double get_cpu_time_us() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec * 1e6 + usage.ru_utime.tv_usec;
}

int main() {
  const char *a_fmt, *b_fmt, *c_fmt, *search;

  // Determine configuration from compile-time flags
#if defined(FORMAT_A_CSR)
  a_fmt = "csr";
#elif defined(FORMAT_A_CSC)
  a_fmt = "csc";
#elif defined(FORMAT_A_COO)
  a_fmt = "coo";
#else
#error "FORMAT_A not defined"
#endif

#if defined(FORMAT_B_CSR)
  b_fmt = "csr";
#elif defined(FORMAT_B_CSC)
  b_fmt = "csc";
#elif defined(FORMAT_B_COO)
  b_fmt = "coo";
#else
#error "FORMAT_B not defined"
#endif

#if defined(FORMAT_C_CSR)
  c_fmt = "csr";
#elif defined(FORMAT_C_CSC)
  c_fmt = "csc";
#elif defined(FORMAT_C_COO)
  c_fmt = "coo";
#else
#error "FORMAT_C not defined"
#endif

#if defined(SEARCH_B)
  search = "B";
#elif defined(SEARCH_C)
  search = "C";
#else
#error "SEARCH not defined"
#endif

  fprintf(stderr, "Hadamard Transpose Benchmark");
#ifdef DEBUG
  fprintf(stderr, " (DEBUG)\n");
#else
  fprintf(stderr, " (FULL)\n");
#endif
  fprintf(stderr, "Configuration: A=%s, B=%s, C=%s, SEARCH=%s\n", a_fmt, b_fmt, c_fmt, search);
  fprintf(stderr, "=============================\n\n");

  // Generate size list
  size_t sizes[NUM_SAMPLES];
  size_t num_sizes;
  generate_sizes(sizes, &num_sizes);

  fprintf(stderr, "Testing sizes: ");
  for (size_t i = 0; i < num_sizes; ++i) {
    fprintf(stderr, "%zu%s", sizes[i], i < num_sizes - 1 ? ", " : "\n");
  }

  fprintf(stderr, "Sparsities: ");
  for (size_t i = 0; i < NUM_SPARSITIES; ++i) {
    fprintf(stderr, "%.2f%s", SPARSITIES[i], i < NUM_SPARSITIES - 1 ? ", " : "\n");
  }

  // Write CSV header to stdout
  printf("A_format,B_format,C_format,search_in,size,B_sparsity,C_sparsity,avg_time_ms\n");

  // Run benchmarks
  for (size_t size_idx = 0; size_idx < num_sizes; ++size_idx) {
    size_t size = sizes[size_idx];
    fprintf(stderr, "Testing size %zu...\n", size);

    for (size_t b_sp_idx = 0; b_sp_idx < NUM_SPARSITIES; ++b_sp_idx) {
      double b_sparsity = SPARSITIES[b_sp_idx];

      for (size_t c_sp_idx = 0; c_sp_idx < NUM_SPARSITIES; ++c_sp_idx) {
        double c_sparsity = SPARSITIES[c_sp_idx];

        // Allocate tensors based on compile-time configuration
        size_t estimated_nnz = (size_t)(size * size * b_sparsity * c_sparsity);
        if (estimated_nnz < 1)
          estimated_nnz = 1;

#if defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSR)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct csr *B = generate_csr(size, size, b_sparsity, SEED);
        struct csr *C = generate_csr(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_CSC)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct csr *B = generate_csr(size, size, b_sparsity, SEED);
        struct csc *C = generate_csc(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_CSR) && defined(FORMAT_C_COO)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct csr *B = generate_csr(size, size, b_sparsity, SEED);
        struct coo *C = generate_coo(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSR)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct coo *B = generate_coo(size, size, b_sparsity, SEED);
        struct csr *C = generate_csr(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_CSC)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct coo *B = generate_coo(size, size, b_sparsity, SEED);
        struct csc *C = generate_csc(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSR) && defined(FORMAT_B_COO) && defined(FORMAT_C_COO)
        struct csr *A = allocate_csr(size, estimated_nnz);
        struct coo *B = generate_coo(size, size, b_sparsity, SEED);
        struct coo *C = generate_coo(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSR)
        struct csc *A = allocate_csc(size, estimated_nnz);
        struct csc *B = generate_csc(size, size, b_sparsity, SEED);
        struct csr *C = generate_csr(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_CSC)
        struct csc *A = allocate_csc(size, estimated_nnz);
        struct csc *B = generate_csc(size, size, b_sparsity, SEED);
        struct csc *C = generate_csc(size, size, c_sparsity, SEED + 1);
#elif defined(FORMAT_A_CSC) && defined(FORMAT_B_CSC) && defined(FORMAT_C_COO)
        struct csc *A = allocate_csc(size, estimated_nnz);
        struct csc *B = generate_csc(size, size, b_sparsity, SEED);
        struct coo *C = generate_coo(size, size, c_sparsity, SEED + 1);
#endif

        // Warmup
        for (int w = 0; w < NUM_WARMUP; ++w) {
          reset_tensor(A);
          hadamard_transpose(A, B, C);
        }

        // Benchmark runs
        double total_time = 0.0;
        for (int r = 0; r < NUM_RUNS; ++r) {
          reset_tensor(A);
          double start = get_cpu_time_us();
          hadamard_transpose(A, B, C);
          double end = get_cpu_time_us();
          total_time += (end - start);
        }

        double avg_time_ms = (total_time / NUM_RUNS) / 1e3;

        // Output CSV line to stdout
        printf("%s,%s,%s,%s,%zu,%.2f,%.2f,%.4f\n", a_fmt, b_fmt, c_fmt, search, size, b_sparsity, c_sparsity,
               avg_time_ms);
        fflush(stdout);

        // Free tensors
        free_tensor(A);
        free_tensor(B);
        free_tensor(C);
      }
    }
  }

  fprintf(stderr, "\nBenchmark complete!\n");
  return 0;
}

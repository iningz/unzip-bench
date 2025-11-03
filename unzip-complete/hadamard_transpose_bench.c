#include "hadamard_transpose.h"
#include "tensor_formats.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

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

// Tensor set for one size and B/C sparsity combination
struct tensor_set {
  struct csr *A_csr;
  struct csc *A_csc;
  struct csr *B_csr;
  struct csc *B_csc;
  struct coo *B_coo;
  struct csr *C_csr;
  struct csc *C_csc;
  struct coo *C_coo;
};

// Generate logarithmically-spaced sizes
void generate_sizes(size_t *sizes, size_t *count) {
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
double get_cpu_time_us() {
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return usage.ru_utime.tv_sec * 1e6 + usage.ru_utime.tv_usec;
}

// Allocate tensor set for given size and B/C sparsities
struct tensor_set *allocate_tensor_set(size_t size, double b_sparsity, double c_sparsity, unsigned int seed) {
  struct tensor_set *ts = malloc(sizeof(struct tensor_set));

  // Allocate output tensors
  size_t estimated_nnz = (size_t)(size * size * b_sparsity * c_sparsity);
  if (estimated_nnz < 1)
    estimated_nnz = 1;
  ts->A_csr = allocate_tensor(csr, size, estimated_nnz);
  ts->A_csc = allocate_tensor(csc, size, estimated_nnz);

  // Generate input tensors with separate sparsities
  ts->B_csr = generate_tensor(csr, size, size, b_sparsity, seed);
  ts->B_csc = generate_tensor(csc, size, size, b_sparsity, seed + 1);
  ts->B_coo = generate_tensor(coo, size, size, b_sparsity, seed + 2);
  ts->C_csr = generate_tensor(csr, size, size, c_sparsity, seed + 3);
  ts->C_csc = generate_tensor(csc, size, size, c_sparsity, seed + 4);
  ts->C_coo = generate_tensor(coo, size, size, c_sparsity, seed + 5);

  return ts;
}

// Free tensor set
void free_tensor_set(struct tensor_set *ts) {
  free_tensor(ts->A_csr);
  free_tensor(ts->A_csc);
  free_tensor(ts->B_csr);
  free_tensor(ts->B_csc);
  free_tensor(ts->B_coo);
  free_tensor(ts->C_csr);
  free_tensor(ts->C_csc);
  free_tensor(ts->C_coo);
  free(ts);
}

// Macro to run a single benchmark - _Generic dispatches based on pointer types
#define BENCH(fp, a_fmt, b_fmt, c_fmt, size, b_sp, c_sp, A, B, C, SEARCH)                                              \
  do {                                                                                                                 \
    for (int w = 0; w < NUM_WARMUP; ++w) {                                                                             \
      reset_tensor(A);                                                                                                 \
      hadamard_transpose(A, B, C, SEARCH);                                                                             \
    }                                                                                                                  \
    double total_time = 0.0;                                                                                           \
    for (int r = 0; r < NUM_RUNS; ++r) {                                                                               \
      reset_tensor(A);                                                                                                 \
      double start = get_cpu_time_us();                                                                                \
      hadamard_transpose(A, B, C, SEARCH);                                                                             \
      double end = get_cpu_time_us();                                                                                  \
      total_time += (end - start);                                                                                     \
    }                                                                                                                  \
    double avg_time_ms = (total_time / NUM_RUNS) / 1e3;                                                                \
    fprintf(fp, "%s,%s,%s,%s,%zu,%.2f,%.2f,%.6f\n", a_fmt, b_fmt, c_fmt, #SEARCH, size, b_sp, c_sp, avg_time_ms);      \
    fflush(fp);                                                                                                        \
  } while (0)

int main() {
  printf("Hadamard Transpose Benchmark");
#ifdef DEBUG
  printf(" (DEBUG)\n");
#else
  printf(" (FULL)\n");
#endif
  printf("=============================\n");

  printf("\n");
  mkdir("results", 0755);
  FILE *fp = fopen("results/hadamard_transpose.csv", "w");
  if (!fp) {
    fprintf(stderr, "Error: Cannot open results/hadamard_transpose.csv for writing\n");
    return 1;
  }

  // Generate size list
  size_t sizes[NUM_SAMPLES];
  size_t num_sizes;
  generate_sizes(sizes, &num_sizes);

  printf("Testing sizes: ");
  for (size_t i = 0; i < num_sizes; ++i) {
    printf("%zu%s", sizes[i], i < num_sizes - 1 ? ", " : "\n");
  }

  printf("Sparsities: ");
  for (size_t i = 0; i < NUM_SPARSITIES; ++i) {
    printf("%.2f%s", SPARSITIES[i], i < NUM_SPARSITIES - 1 ? ", " : "\n");
  }

  // Write CSV header
  fprintf(fp, "A_format,B_format,C_format,search_in,size,B_sparsity,C_sparsity,avg_time_us\n");

  // Run benchmarks
  for (size_t size_idx = 0; size_idx < num_sizes; ++size_idx) {
    size_t size = sizes[size_idx];
    printf("Testing size %zu...\n", size);

    for (size_t b_sp_idx = 0; b_sp_idx < NUM_SPARSITIES; ++b_sp_idx) {
      double b_sparsity = SPARSITIES[b_sp_idx];

      for (size_t c_sp_idx = 0; c_sp_idx < NUM_SPARSITIES; ++c_sp_idx) {
        double c_sparsity = SPARSITIES[c_sp_idx];

        // Allocate tensor set for this size/B_sparsity/C_sparsity combination
        struct tensor_set *ts = allocate_tensor_set(size, b_sparsity, c_sparsity, SEED);

        // Run all configs
        BENCH(fp, "csr", "csr", "csr", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_csr, ts->C_csr, snd);
        BENCH(fp, "csr", "csr", "csc", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_csr, ts->C_csc, snd);
        BENCH(fp, "csr", "csr", "coo", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_csr, ts->C_coo, snd);
        BENCH(fp, "csr", "coo", "csr", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_coo, ts->C_csr, snd);
        BENCH(fp, "csr", "coo", "csc", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_coo, ts->C_csc, snd);
        BENCH(fp, "csr", "coo", "coo", size, b_sparsity, c_sparsity, ts->A_csr, ts->B_coo, ts->C_coo, snd);
        BENCH(fp, "csc", "csc", "csr", size, b_sparsity, c_sparsity, ts->A_csc, ts->B_csc, ts->C_csr, snd);
        BENCH(fp, "csc", "csc", "csc", size, b_sparsity, c_sparsity, ts->A_csc, ts->B_csc, ts->C_csc, snd);
        BENCH(fp, "csc", "csc", "coo", size, b_sparsity, c_sparsity, ts->A_csc, ts->B_csc, ts->C_coo, snd);

        free_tensor_set(ts);
      }
    }
  }

  fclose(fp);
  printf("\nBenchmark complete! Results saved to results/hadamard_transpose.csv\n");
  return 0;
}

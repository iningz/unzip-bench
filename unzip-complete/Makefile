# Compiler and flags
CC = cc
CFLAGS = -Wall -Wextra
OPTFLAGS = -O3 -march=native -flto -funroll-loops -DNDEBUG
LIBS = -lm

# CPU affinity
CPU_CORES = 0-7

# Source files
KERNEL_SRC = hadamard_transpose.c
UTIL_SRC = tensor_formats.c
TEST_SRC = hadamard_transpose_test.c
BENCH_SRC = hadamard_transpose_bench.c
HEADERS = hadamard_transpose.h tensor_formats.h

# Directories
BUILD_DIR = build
RESULTS_DIR = results

# Configuration variants to build
CONFIGS = \
	csr_csr_csr_c \
	csr_csr_csr_b \
    csr_csr_csc_c \
	csr_csr_csc_b \
	csr_csr_coo_c \
	csr_coo_csr_c \
	csr_coo_csc_c \
	csr_coo_coo_c \
	csc_csc_csr_c \
	csc_csc_csc_c \
	csc_csc_coo_c

# =============================================================================
# Build rules
# =============================================================================

$(BUILD_DIR)/test_%: $(KERNEL_SRC) $(UTIL_SRC) $(TEST_SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(eval PARTS := $(subst _, ,$*))
	$(eval A_FMT := $(word 1,$(PARTS)))
	$(eval B_FMT := $(word 2,$(PARTS)))
	$(eval C_FMT := $(word 3,$(PARTS)))
	$(eval SEARCH := $(word 4,$(PARTS)))
	@echo "Building test: A=$(A_FMT), B=$(B_FMT), C=$(C_FMT), SEARCH=$(SEARCH)"
	$(CC) $(CFLAGS) \
		-DFORMAT_A_$(shell echo $(A_FMT) | tr a-z A-Z) \
		-DFORMAT_B_$(shell echo $(B_FMT) | tr a-z A-Z) \
		-DFORMAT_C_$(shell echo $(C_FMT) | tr a-z A-Z) \
		-DSEARCH_$(shell echo $(SEARCH) | tr a-z A-Z) \
		-o $@ $(KERNEL_SRC) $(UTIL_SRC) $(TEST_SRC) $(LIBS)

$(BUILD_DIR)/bench_debug_%: $(KERNEL_SRC) $(UTIL_SRC) $(BENCH_SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(eval PARTS := $(subst _, ,$*))
	$(eval A_FMT := $(word 1,$(PARTS)))
	$(eval B_FMT := $(word 2,$(PARTS)))
	$(eval C_FMT := $(word 3,$(PARTS)))
	$(eval SEARCH := $(word 4,$(PARTS)))
	@echo "Building benchmark (DEBUG): A=$(A_FMT), B=$(B_FMT), C=$(C_FMT), SEARCH=$(SEARCH)"
	$(CC) $(CFLAGS) $(OPTFLAGS) -DDEBUG \
		-DFORMAT_A_$(shell echo $(A_FMT) | tr a-z A-Z) \
		-DFORMAT_B_$(shell echo $(B_FMT) | tr a-z A-Z) \
		-DFORMAT_C_$(shell echo $(C_FMT) | tr a-z A-Z) \
		-DSEARCH_$(shell echo $(SEARCH) | tr a-z A-Z) \
		-o $@ $(KERNEL_SRC) $(UTIL_SRC) $(BENCH_SRC) $(LIBS)

$(BUILD_DIR)/bench_%: $(KERNEL_SRC) $(UTIL_SRC) $(BENCH_SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(eval PARTS := $(subst _, ,$*))
	$(eval A_FMT := $(word 1,$(PARTS)))
	$(eval B_FMT := $(word 2,$(PARTS)))
	$(eval C_FMT := $(word 3,$(PARTS)))
	$(eval SEARCH := $(word 4,$(PARTS)))
	@echo "Building benchmark (FULL): A=$(A_FMT), B=$(B_FMT), C=$(C_FMT), SEARCH=$(SEARCH)"
	$(CC) $(CFLAGS) $(OPTFLAGS) \
		-DFORMAT_A_$(shell echo $(A_FMT) | tr a-z A-Z) \
		-DFORMAT_B_$(shell echo $(B_FMT) | tr a-z A-Z) \
		-DFORMAT_C_$(shell echo $(C_FMT) | tr a-z A-Z) \
		-DSEARCH_$(shell echo $(SEARCH) | tr a-z A-Z) \
		-o $@ $(KERNEL_SRC) $(UTIL_SRC) $(BENCH_SRC) $(LIBS)

# =============================================================================
# Default target
# =============================================================================

.PHONY: all
all: test

# =============================================================================
# Build targets
# =============================================================================

.PHONY: build
build: build-test build-bench-debug build-bench

.PHONY: build-test
build-test: $(patsubst %,$(BUILD_DIR)/test_%, $(CONFIGS))

.PHONY: build-test-%
build-test-%: $(BUILD_DIR)/test_%
	@echo "Built test binary: $(BUILD_DIR)/test_$*"

.PHONY: build-bench-debug
build-bench-debug: $(patsubst %,$(BUILD_DIR)/bench_debug_%, $(CONFIGS))

.PHONY: build-bench-debug-%
build-bench-debug-%: $(BUILD_DIR)/bench_debug_%
	@echo "Built debug benchmark binary: $(BUILD_DIR)/bench_debug_$*"

.PHONY: build-bench
build-bench: $(patsubst %,$(BUILD_DIR)/bench_%, $(CONFIGS))

.PHONY: build-bench-%
build-bench-%: $(BUILD_DIR)/bench_%
	@echo "Built benchmark binary: $(BUILD_DIR)/bench_$*"

# =============================================================================
# Test targets
# =============================================================================

.PHONY: test
test: build-test
	@$(MAKE) $(patsubst %,test-%, $(CONFIGS))

.PHONY: test-%
test-%: $(BUILD_DIR)/test_%
	@echo "Running test: $*"
	@$(BUILD_DIR)/test_$*

# =============================================================================
# Benchmark targets (DEBUG mode - fast iteration)
# =============================================================================

.PHONY: bench-debug
bench-debug: build-bench-debug
	@$(MAKE) $(patsubst %,bench-debug-%, $(CONFIGS))

.PHONY: bench-debug-%
bench-debug-%: $(BUILD_DIR)/bench_debug_%
	@echo "Running benchmark (DEBUG): $*"
	@mkdir -p $(RESULTS_DIR)
	@if command -v taskset >/dev/null 2>&1; then \
		taskset -c $(CPU_CORES) $(BUILD_DIR)/bench_debug_$* > $(RESULTS_DIR)/debug_$*.csv; \
	else \
		$(BUILD_DIR)/bench_debug_$* > $(RESULTS_DIR)/debug_$*.csv; \
	fi
	@echo "Results saved to $(RESULTS_DIR)/debug_$*.csv"

# =============================================================================
# Benchmark targets (FULL mode)
# =============================================================================

.PHONY: bench
bench: build-bench
	@$(MAKE) $(patsubst %,bench-%, $(CONFIGS))

.PHONY: bench-%
bench-%: $(BUILD_DIR)/bench_%
	@echo "Running benchmark (FULL): $*"
	@mkdir -p $(RESULTS_DIR)
	@if command -v taskset >/dev/null 2>&1; then \
		taskset -c $(CPU_CORES) $(BUILD_DIR)/bench_$* > $(RESULTS_DIR)/$*.csv; \
	else \
		$(BUILD_DIR)/bench_$* > $(RESULTS_DIR)/$*.csv; \
	fi
	@echo "Results saved to $(RESULTS_DIR)/$*.csv"

# =============================================================================
# Clean targets
# =============================================================================

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(RESULTS_DIR)

.PHONY: clean-build
clean-build:
	rm -rf $(BUILD_DIR)

.PHONY: clean-results
clean-results:
	rm -rf $(RESULTS_DIR)

# =============================================================================
# Help target
# =============================================================================

.PHONY: help
help:
	@echo "Hadamard Transpose Benchmark System"
	@echo "===================================="
	@echo "  make build                       - Build all binaries"
	@echo "  make build-test                  - Build all test binaries"
	@echo "  make build-test-<config>         - Build a specific test binary"
	@echo "  make build-bench                 - Build all benchmark binaries"
	@echo "  make build-bench-<config>        - Build a specific benchmark binary"
	@echo "  make build-bench-debug           - Build all debug benchmark binaries"
	@echo "  make build-bench-debug-<config>  - Build a specific debug benchmark binary"
	@echo "  make test                        - Build and run all tests"
	@echo "  make test-<config>               - Run a specific test"
	@echo "  make bench                       - Build and run all benchmarks"
	@echo "  make bench-<config>              - Run one benchmark"
	@echo "  make bench-debug                 - Build and run all debug benchmarks"
	@echo "  make bench-debug-<config>        - Run one debug benchmark"
	@echo "  make clean                       - Remove build/ and results/"
	@echo "  make clean-build                 - Remove build/ only"
	@echo "  make clean-results               - Remove results/ only"
	@echo ""
	@echo "Available configurations:"
	@for config in $(CONFIGS); do echo "  $$config"; done
	@echo ""

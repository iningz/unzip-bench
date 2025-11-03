# Compiler and flags
CC = cc
CFLAGS = -Wall -Wextra
OPTFLAGS = -O3 -march=native -flto -funroll-loops -DNDEBUG
LIBS = -lm

# CPU affinity
CPU_CORES = 0-7

# Source files
SOURCES = hadamard_transpose.c tensor_formats.c
HEADERS = hadamard_transpose.h tensor_formats.h
TEST_SRC = hadamard_transpose_test.c
BENCH_SRC = hadamard_transpose_bench.c

# Output binaries
TEST_BIN = hadamard_transpose_test
BENCH_DEBUG_BIN = hadamard_transpose_bench_debug
BENCH_FULL_BIN = hadamard_transpose_bench

# Results directory
RESULTS_DIR = results

# Default target
.PHONY: all
all: test bench

# Build and run tests
.PHONY: test
test: $(TEST_BIN)
	@./$(TEST_BIN)

$(TEST_BIN): $(SOURCES) $(TEST_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(TEST_SRC) $(LIBS)

# Build and run benchmark in debug mode (fast iteration)
.PHONY: bench-debug
bench-debug: $(BENCH_DEBUG_BIN)
	@mkdir -p $(RESULTS_DIR)
	@if command -v taskset >/dev/null 2>&1; then \
		taskset -c $(CPU_CORES) ./$(BENCH_DEBUG_BIN); \
	else \
		echo "Warning: taskset not found. Running without CPU affinity pinning."; \
		./$(BENCH_DEBUG_BIN); \
	fi

$(BENCH_DEBUG_BIN): $(SOURCES) $(BENCH_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -DDEBUG -o $@ $(SOURCES) $(BENCH_SRC) $(LIBS)

# Build and run benchmark in full mode (full benchmark)
.PHONY: bench
bench: $(BENCH_FULL_BIN)
	@mkdir -p $(RESULTS_DIR)
	@if command -v taskset >/dev/null 2>&1; then \
		taskset -c $(CPU_CORES) ./$(BENCH_FULL_BIN); \
	else \
		echo "Warning: taskset not found. Running without CPU affinity pinning."; \
		./$(BENCH_FULL_BIN); \
	fi

$(BENCH_FULL_BIN): $(SOURCES) $(BENCH_SRC) $(HEADERS)
	$(CC) $(CFLAGS) $(OPTFLAGS) -o $@ $(SOURCES) $(BENCH_SRC) $(LIBS)

# Build only (no execution)
.PHONY: build
build: $(TEST_BIN) $(BENCH_DEBUG_BIN) $(BENCH_FULL_BIN)

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(TEST_BIN) $(BENCH_DEBUG_BIN) $(BENCH_FULL_BIN)
	rm -rf $(RESULTS_DIR)

# Clean only binaries, keep results
.PHONY: clean-bin
clean-bin:
	rm -f $(TEST_BIN) $(BENCH_DEBUG_BIN) $(BENCH_FULL_BIN)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make test           - Build and run tests"
	@echo "  make bench          - Build and run benchmark in FULL mode on big cores"
	@echo "  make bench-debug    - Build and run benchmark in DEBUG mode on big cores"
	@echo "  make build          - Build all binaries without running"
	@echo "  make clean          - Remove all build artifacts and results"
	@echo "  make clean-bin      - Remove binaries only, keep results"
	@echo "  make all            - Run test and bench"
	@echo "  make help           - Show this help message"

CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = shared_memory_basic matrix_tiling reduction_shared bank_conflicts

all: $(TARGETS)

$(BUILD_DIR)/shared_memory_basic: shared_memory_basic.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/shared_memory_basic shared_memory_basic.cu

$(BUILD_DIR)/matrix_tiling: matrix_tiling.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_tiling matrix_tiling.cu

$(BUILD_DIR)/reduction_shared: reduction_shared.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/reduction_shared reduction_shared.cu

$(BUILD_DIR)/bank_conflicts: bank_conflicts.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/bank_conflicts bank_conflicts.cu

shared_memory_basic: $(BUILD_DIR)/shared_memory_basic
matrix_tiling: $(BUILD_DIR)/matrix_tiling
reduction_shared: $(BUILD_DIR)/reduction_shared
bank_conflicts: $(BUILD_DIR)/bank_conflicts

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/shared_memory_basic $(BUILD_DIR)/matrix_tiling \
	      $(BUILD_DIR)/reduction_shared $(BUILD_DIR)/bank_conflicts

.PHONY: all clean shared_memory_basic matrix_tiling reduction_shared bank_conflicts

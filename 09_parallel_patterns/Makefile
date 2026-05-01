CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = reduction scan_prefix_sum reduction_warp_optimized

all: $(TARGETS)

$(BUILD_DIR)/reduction: reduction.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/reduction reduction.cu

$(BUILD_DIR)/scan_prefix_sum: scan_prefix_sum.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/scan_prefix_sum scan_prefix_sum.cu

$(BUILD_DIR)/reduction_warp_optimized: reduction_warp_optimized.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/reduction_warp_optimized reduction_warp_optimized.cu

reduction: $(BUILD_DIR)/reduction
scan_prefix_sum: $(BUILD_DIR)/scan_prefix_sum
reduction_warp_optimized: $(BUILD_DIR)/reduction_warp_optimized

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/reduction $(BUILD_DIR)/scan_prefix_sum $(BUILD_DIR)/reduction_warp_optimized

.PHONY: all clean reduction scan_prefix_sum reduction_warp_optimized

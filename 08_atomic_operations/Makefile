CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = atomic_add atomic_compare histogram

all: $(TARGETS)

$(BUILD_DIR)/atomic_add: atomic_add.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/atomic_add atomic_add.cu

$(BUILD_DIR)/atomic_compare: atomic_compare.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/atomic_compare atomic_compare.cu

$(BUILD_DIR)/histogram: histogram.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/histogram histogram.cu

atomic_add: $(BUILD_DIR)/atomic_add
atomic_compare: $(BUILD_DIR)/atomic_compare
histogram: $(BUILD_DIR)/histogram

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/atomic_add $(BUILD_DIR)/atomic_compare $(BUILD_DIR)/histogram

.PHONY: all clean atomic_add atomic_compare histogram

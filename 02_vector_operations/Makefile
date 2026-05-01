CC = nvcc
# Use -arch=sm_89 for auto-detection (works with RTX 5070 Ti CC 12.0)
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = vector_scale vector_dot vector_operations_fused

all: $(TARGETS)

$(BUILD_DIR)/vector_scale: vector_scale.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_scale vector_scale.cu

$(BUILD_DIR)/vector_dot: vector_dot.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_dot vector_dot.cu

$(BUILD_DIR)/vector_operations_fused: vector_operations_fused.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_operations_fused vector_operations_fused.cu

vector_scale: $(BUILD_DIR)/vector_scale
vector_dot: $(BUILD_DIR)/vector_dot
vector_operations_fused: $(BUILD_DIR)/vector_operations_fused

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/vector_scale $(BUILD_DIR)/vector_dot $(BUILD_DIR)/vector_operations_fused

.PHONY: all clean vector_scale vector_dot vector_operations_fused

CC = nvcc
# Use -arch=sm_89 for auto-detection (works with RTX 5070 Ti CC 12.0)
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = matrix_add matrix_multiply matrix_transpose matrix_multiply_optimized_ada matrix_multiply_tensor_core

all: $(TARGETS)

$(BUILD_DIR)/matrix_add: matrix_add.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_add matrix_add.cu

$(BUILD_DIR)/matrix_multiply: matrix_multiply.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_multiply matrix_multiply.cu

$(BUILD_DIR)/matrix_transpose: matrix_transpose.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_transpose matrix_transpose.cu

$(BUILD_DIR)/matrix_multiply_optimized_ada: matrix_multiply_optimized_ada.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_multiply_optimized_ada matrix_multiply_optimized_ada.cu

$(BUILD_DIR)/matrix_multiply_tensor_core: matrix_multiply_tensor_core.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/matrix_multiply_tensor_core matrix_multiply_tensor_core.cu

matrix_add: $(BUILD_DIR)/matrix_add
matrix_multiply: $(BUILD_DIR)/matrix_multiply
matrix_transpose: $(BUILD_DIR)/matrix_transpose
matrix_multiply_optimized_ada: $(BUILD_DIR)/matrix_multiply_optimized_ada
matrix_multiply_tensor_core: $(BUILD_DIR)/matrix_multiply_tensor_core

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/matrix_add $(BUILD_DIR)/matrix_multiply $(BUILD_DIR)/matrix_transpose $(BUILD_DIR)/matrix_multiply_optimized_ada $(BUILD_DIR)/matrix_multiply_tensor_core

.PHONY: all clean matrix_add matrix_multiply matrix_transpose matrix_multiply_optimized_ada matrix_multiply_tensor_core

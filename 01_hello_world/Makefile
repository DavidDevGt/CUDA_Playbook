CC = nvcc
# Use -arch=sm_89 for auto-detection (works with RTX 5070 Ti CC 12.0)
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = hello vector_add vector_add_unified

all: $(TARGETS)

$(BUILD_DIR)/hello: hello.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/hello hello.cu

$(BUILD_DIR)/vector_add: vector_add.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_add vector_add.cu

$(BUILD_DIR)/vector_add_unified: vector_add_unified.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_add_unified vector_add_unified.cu

hello: $(BUILD_DIR)/hello
vector_add: $(BUILD_DIR)/vector_add
vector_add_unified: $(BUILD_DIR)/vector_add_unified

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/hello $(BUILD_DIR)/vector_add $(BUILD_DIR)/vector_add_unified

.PHONY: all clean hello vector_add vector_add_unified

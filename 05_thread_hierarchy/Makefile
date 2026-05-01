CC = nvcc
# Updated for Blackwell (RTX 5070 Ti): -arch=sm_89 for CC 12.0
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = block_dimensions grid_stride indexing_1d indexing_2d indexing_3d

all: $(TARGETS)

$(BUILD_DIR)/block_dimensions: block_dimensions.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/block_dimensions block_dimensions.cu

$(BUILD_DIR)/grid_stride: grid_stride.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/grid_stride grid_stride.cu

$(BUILD_DIR)/indexing_1d: indexing_1d.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/indexing_1d indexing_1d.cu

$(BUILD_DIR)/indexing_2d: indexing_2d.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/indexing_2d indexing_2d.cu

$(BUILD_DIR)/indexing_3d: indexing_3d.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/indexing_3d indexing_3d.cu

block_dimensions: $(BUILD_DIR)/block_dimensions
grid_stride: $(BUILD_DIR)/grid_stride
indexing_1d: $(BUILD_DIR)/indexing_1d
indexing_2d: $(BUILD_DIR)/indexing_2d
indexing_3d: $(BUILD_DIR)/indexing_3d

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/block_dimensions $(BUILD_DIR)/grid_stride \
	      $(BUILD_DIR)/indexing_1d $(BUILD_DIR)/indexing_2d $(BUILD_DIR)/indexing_3d

.PHONY: all clean block_dimensions grid_stride indexing_1d indexing_2d indexing_3d

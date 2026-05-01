CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = vector_max image_convolution

all: $(TARGETS)

$(BUILD_DIR)/vector_max: vector_max.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/vector_max vector_max.cu

$(BUILD_DIR)/image_convolution: image_convolution.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/image_convolution image_convolution.cu

vector_max: $(BUILD_DIR)/vector_max
image_convolution: $(BUILD_DIR)/image_convolution

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/vector_max $(BUILD_DIR)/image_convolution

.PHONY: all clean vector_max image_convolution

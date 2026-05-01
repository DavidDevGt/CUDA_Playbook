CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = check_cuda device_info

all: $(TARGETS)

$(BUILD_DIR)/check_cuda: check_cuda.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/check_cuda check_cuda.cu

$(BUILD_DIR)/device_info: device_info.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/device_info device_info.cu

check_cuda: $(BUILD_DIR)/check_cuda
device_info: $(BUILD_DIR)/device_info

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/check_cuda $(BUILD_DIR)/device_info

.PHONY: all clean check_cuda device_info

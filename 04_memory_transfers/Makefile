CC = nvcc
# Use -arch=sm_89 for auto-detection (works with RTX 5070 Ti CC 12.0)
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = host_to_device device_to_host pinned_memory zero_copy async_copy_example

all: $(TARGETS)

$(BUILD_DIR)/host_to_device: host_to_device.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/host_to_device host_to_device.cu

$(BUILD_DIR)/device_to_host: device_to_host.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/device_to_host device_to_host.cu

$(BUILD_DIR)/pinned_memory: pinned_memory.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/pinned_memory pinned_memory.cu

$(BUILD_DIR)/zero_copy: zero_copy.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/zero_copy zero_copy.cu

$(BUILD_DIR)/async_copy_example: async_copy_example.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/async_copy_example async_copy_example.cu

host_to_device: $(BUILD_DIR)/host_to_device
device_to_host: $(BUILD_DIR)/device_to_host
pinned_memory: $(BUILD_DIR)/pinned_memory
zero_copy: $(BUILD_DIR)/zero_copy
async_copy_example: $(BUILD_DIR)/async_copy_example

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/host_to_device $(BUILD_DIR)/device_to_host $(BUILD_DIR)/pinned_memory $(BUILD_DIR)/zero_copy $(BUILD_DIR)/async_copy_example

.PHONY: all clean host_to_device device_to_host pinned_memory zero_copy async_copy_example

CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = occupancy memory_coalescing loop_unrolling profile_events

all: $(TARGETS)

$(BUILD_DIR)/occupancy: occupancy.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/occupancy occupancy.cu

$(BUILD_DIR)/memory_coalescing: memory_coalescing.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/memory_coalescing memory_coalescing.cu

$(BUILD_DIR)/loop_unrolling: loop_unrolling.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/loop_unrolling loop_unrolling.cu

$(BUILD_DIR)/profile_events: profile_events.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/profile_events profile_events.cu

occupancy: $(BUILD_DIR)/occupancy
memory_coalescing: $(BUILD_DIR)/memory_coalescing
loop_unrolling: $(BUILD_DIR)/loop_unrolling
profile_events: $(BUILD_DIR)/profile_events

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/occupancy $(BUILD_DIR)/memory_coalescing \
	      $(BUILD_DIR)/loop_unrolling $(BUILD_DIR)/profile_events

.PHONY: all clean occupancy memory_coalescing loop_unrolling profile_events

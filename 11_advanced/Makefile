CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = streams events dynamic_parallelism

all: $(TARGETS)

$(BUILD_DIR)/streams: streams.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/streams streams.cu

$(BUILD_DIR)/events: events.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/events events.cu

$(BUILD_DIR)/dynamic_parallelism: dynamic_parallelism.cu
	$(CC) $(CFLAGS) $(INCLUDES) -rdc=true -lcudadevrt -o $(BUILD_DIR)/dynamic_parallelism dynamic_parallelism.cu

streams: $(BUILD_DIR)/streams
events: $(BUILD_DIR)/events
dynamic_parallelism: $(BUILD_DIR)/dynamic_parallelism

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/streams $(BUILD_DIR)/events $(BUILD_DIR)/dynamic_parallelism

.PHONY: all clean streams events dynamic_parallelism

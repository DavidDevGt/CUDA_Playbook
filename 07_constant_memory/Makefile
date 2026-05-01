CC = nvcc
# Updated for Ada Lovelace (RTX 5070 Ti): -arch=sm_89 replaces sm_70
CFLAGS = -O2 -arch=sm_89
INCLUDES = -I../common
BUILD_DIR = ../build

TARGETS = constant_array lookup_table

all: $(TARGETS)

$(BUILD_DIR)/constant_array: constant_array.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/constant_array constant_array.cu

$(BUILD_DIR)/lookup_table: lookup_table.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BUILD_DIR)/lookup_table lookup_table.cu

constant_array: $(BUILD_DIR)/constant_array
lookup_table: $(BUILD_DIR)/lookup_table

$(BUILD_DIR)/%: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(BUILD_DIR)/constant_array $(BUILD_DIR)/lookup_table

.PHONY: all clean constant_array lookup_table

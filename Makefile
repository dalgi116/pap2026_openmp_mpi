# --- Compiler & Dependencies ---
CXX          := mpiCC
OPENCV_FLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS  := $(shell pkg-config --libs opencv4)

CXXFLAGS += -g -std=gnu++17 -pedantic -Wall -fopenmp $(subst -I,-isystem ,$(OPENCV_FLAGS))
LDFLAGS  += -fopenmp
LDLIBS   += -lmpi $(OPENCV_LIBS)

# --- Project & Runtime Settings ---
BUILD_DIR   := build
TARGET      := $(BUILD_DIR)/main
SRC_DIR     := test/src
OUT_DIR     := test/out

OMP_THREADS := 4
MPI_NP      := 3
# Disable automatic core binding so OpenMP threads can spread across cores.
# https://docs.open-mpi.org/en/v5.0.x/man-openmpi/man1/mpirun.1.html
MPI_OPTIONS := --bind-to none --host localhost,localhost,localhost

# --- Build Rules ---
.PHONY: all run dep depend clean

all: $(TARGET)

$(TARGET): $(BUILD_DIR)/main.o
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/main.o: src/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

# --- Run & Maintenance ---
run: $(TARGET)
	OMP_NUM_THREADS=$(OMP_THREADS) mpirun $(MPI_OPTIONS) -np $(MPI_NP) -x OMP_NUM_THREADS ./$(TARGET) -src $(SRC_DIR) -out $(OUT_DIR)

dep: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -w -E -M src/*.cpp > $(BUILD_DIR)/depend

depend: $(BUILD_DIR)
	@touch $(BUILD_DIR)/depend

clean:
	rm -rf $(BUILD_DIR)

-include $(BUILD_DIR)/depend

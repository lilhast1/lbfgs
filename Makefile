# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 --use_fast_math -arch=sm_61
INCLUDES = -Isrc -Isrc/lbfgs -Isrc/utils
LIBS = -lcublas

# Source files
CUDA_SOURCES = src/lbfgs/lbfgs.cu \
               src/lbfgs/lbfgs_kernel.cu \
               src/lbfgs/lbfgs_lnsrch.cu

CPP_SOURCES = src/main.cpp

# Output
TARGET = lbfgs_cuda

# Build rule
all: $(TARGET)

$(TARGET): $(CUDA_SOURCES) $(CPP_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $(TARGET) $(CUDA_SOURCES) $(CPP_SOURCES) $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)

# Run rule
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
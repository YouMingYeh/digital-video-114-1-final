# Compilers
NVCC = nvcc

# Paths
LODEPNG_DIR = lodepng
GLM_DIR = glm

# Compiler flags
CPPFLAGS = -I$(LODEPNG_DIR) -I$(GLM_DIR)/include
NVCCFLAGS = -std=c++17 -O3 -arch=sm_86 -Xptxas=-v --use_fast_math

# Source files
LODEPNG = $(LODEPNG_DIR)/lodepng.cpp

# Target
TARGET = mandelbulb

all: $(TARGET)

$(TARGET): mandelbulb.cu $(LODEPNG)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) mandelbulb.cu $(LODEPNG) -o $(TARGET)

# Clean
clean:
	rm -f $(TARGET)

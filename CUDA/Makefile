NVCC = nvcc
GCC = g++
NVCC_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3 --expt-relaxed-constexpr
GCC_FLAGS = -c -O3
LINK_FLAGS = -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'

OBJS = main.o ycbcr.o JPEGCompress.o
TARGET = JPEGCompress_main
IMG = output.png

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(LINK_FLAGS) -o $@ $^

ycbcr.o: ycbcr.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

JPEGCompress.o: JPEGCompress.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<	

main.o: main.cpp
	$(GCC) $(GCC_FLAGS) -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET) $(IMG)
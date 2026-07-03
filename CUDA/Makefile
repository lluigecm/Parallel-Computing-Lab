NVCC   = nvcc
# sm_75 = arquitetura Turing (GTX 1650). Ajuste se rodar em outra GPU.
NVFLAGS = -O2 -arch=sm_75

all: cuda

cuda: cuda.cu gaussian.cu gaussian.cuh
	$(NVCC) $(NVFLAGS) -o cuda cuda.cu gaussian.cu

clean:
	rm -f cuda output_cuda_*.png

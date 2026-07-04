NVCC   = nvcc
# sm_75 = arquitetura Turing (GTX 1650). Ajuste se rodar em outra GPU.
# -U_GNU_SOURCE: evita conflito do glibc novo (2.43+) nas extensoes rsqrt/cospi/sinpi
# declaradas com noexcept diferente do que o CUDA math_functions.h espera.
NVFLAGS = -O2 -arch=sm_75 -Xcompiler -U_GNU_SOURCE

all: cuda

cuda: cuda.cu gaussian.cu gaussian.cuh
	$(NVCC) $(NVFLAGS) -o cuda cuda.cu gaussian.cu

clean:
	rm -f cuda output_cuda_*.png

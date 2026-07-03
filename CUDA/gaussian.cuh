#ifndef GAUSSIAN_CUH
#define GAUSSIAN_CUH

float* alloc_img(int height, int width);

// kernel_size deve ser impar e >= 3
// retorna array 1D alocado dinamicamente (caller faz free)
float* generate_kernel(int kernel_size, float sigma);

float* load_img(const char* filename, int* width, int* height);
void   save_img(const char* filename, float* img, int height, int width);

#endif

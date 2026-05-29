#ifndef GAUSSIAN_H
#define GAUSSIAN_H

float* alloc_img(int height, int width);

// kernel_size deve ser impar e >= 3
// retorna array alocado dinamicamente, caller faz free()
float* generate_kernel(int kernel_size, float sigma);

float* load_img(const char* filename, int* width, int* height);
void   save_img(const char* filename, float* img, int height, int width);

// Versao local do filtro gaussiano para usar com MPI
// src e dst tem tamanho (local_rows + 2*half) * width
// first=1 se esse processo tem a primeira linha da imagem, last=1 se tem a ultima
void gaussian_blur_local(float* src, float* dst, float* kernel, int kernel_size,
                          int local_rows, int width, int first, int last);

#endif

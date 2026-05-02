#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <math.h>

#define KERNEL_SIZE 3

// Alocação e inicialização
float* alloc_img(int height, int width);
void   init_img(float* img, int height, int width);

// Geração do kernel gaussiano a partir do sigma
void generate_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE], float sigma);

// Carrega PNG em escala de cinza → array de floats
float* load_img(const char* filename, int* width, int* height);

// Salva array de floats → PNG em escala de cinza
void save_img(const char* filename, float* img, int height, int width);

// Versões do filtro gaussiano
void gaussian_blur_serial (float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width);
void gaussian_blur_openmp (float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width);
void gaussian_blur_tiling (float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width);

#endif // GAUSSIAN_H

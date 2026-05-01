#ifndef GAUSSIAN_H
#define GAUSSIAN_H


#define KERNEL_SIZE 3


// Aloca array 1d para armazenar a imagem
float* alloc_img(int height, int width);

// Preenche a imagem com valores aleatorios
void init_img(float* img, int height, int width);

// Define os pesos do kernel
void define_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE]);

// Aplica UMA iteracao do filtro gaussiano
void gaussian_blur(float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width);

#endif // GAUSSIAN_H
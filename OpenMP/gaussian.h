#ifndef GAUSSIAN_H
#define GAUSSIAN_H




// Aloca array 1d para armazenar a imagem
float* alloc_img(int height, int width);

// Preenche a imagem com valores aleatorios
void init_img(float* img, int height, int width);

// Aplica UMA iteracao do filtro gaussiano
void gaussian_blur_serial(float* src, float* dst, int height, int width);

void gaussian_blur_openmp(float* src, float* dst, int height, int width);


#endif // GAUSSIAN_H
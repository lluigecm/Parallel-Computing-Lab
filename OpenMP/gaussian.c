#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "gaussian.h"


static const float KERNEL[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

#define KERNEL_SUM 16.0f


float* alloc_img(int height, int width) {
    float* img = (float*)malloc(height * width * sizeof(float));

    if (img == NULL) {
        fprintf(stderr, "Erro: falha ao alocar imagem\n");
        exit(1);
    }
    return img;
}


void init_img(float* img, int height, int width) {
    for (int i = 0; i < height * width; i++) {
        img[i] = rand() % 256; // Valores entre 0 e 255
    }
}


void gaussian_blur_serial(float* src, float* dst, int height, int width) {

    // Copia as bordas sem alteração
    for (int i = 0; i < width; i++) {
        dst[0 * width + i] = src[0 * width + i]; // Copia bordas

        dst[(height - 1) * width + i] = src[(height - 1) * width + i];
    }

    for (int j = 0; j < height; j++) {
        dst[j * width + 0] = src[j * width + 0]; // Copia bordas

        dst[j * width + (width - 1)] = src[j * width + (width - 1)];
    }

    // Aplica o filtro gaussiano para os pixels internos
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0f;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum += src[(i + k) * width + (j + l)] * KERNEL[k + 1][l + 1];
                }
            }
            dst[i * width + j] = sum / KERNEL_SUM;
        }
    }
}


void gaussian_blur_openmp(float* src, float* dst, int height, int width) {

    // Copia bordas sem filtrar
    for (int j = 0; j < width; j++) {
        dst[0 * width + j]            = src[0 * width + j];           // linha topo
        dst[(height-1) * width + j]   = src[(height-1) * width + j];  // linha base
    }

     for (int i = 0; i < height; i++) {
        dst[i * width + 0]            = src[i * width + 0];           // coluna esquerda
        dst[i * width + (width-1)]    = src[i * width + (width-1)];   // coluna direita
    }

    // Aplica o filtro gaussiano para os pixels internos
    #pragma omp parallel for
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0f;
            for (int k = -1; k <= 1; k++) {
                for (int l = -1; l <= 1; l++) {
                    sum += src[(i + k) * width + (j + l)] * KERNEL[k + 1][l + 1];
                }
            }
            dst[i * width + j] = sum / KERNEL_SUM;
        }
    }
}

void gaussian_blur_tiling(float* src, float* dst, int height, int width) {
    // Tamanho do bloco — ajustável
    int TILE = 128;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ti = 1; ti < height - 1; ti += TILE) {
        for (int tj = 1; tj < width - 1; tj += TILE) {

            // Limites do bloco atual
            int i_end = ti + TILE < height - 1 ? ti + TILE : height - 1;
            int j_end = tj + TILE < width - 1  ? tj + TILE : width - 1;

            // Processa os pixels dentro do bloco
            for (int i = ti; i < i_end; i++) {
                for (int j = tj; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int k = -1; k <= 1; k++)
                        for (int l = -1; l <= 1; l++)
                            sum += src[(i+k) * width + (j+l)] * KERNEL[k+1][l+1];
                    dst[i * width + j] = sum / KERNEL_SUM;
                }
            }
        }
    }

    // Copia bordas
    for (int j = 0; j < width; j++) {
        dst[0 * width + j]           = src[0 * width + j];
        dst[(height-1) * width + j]  = src[(height-1) * width + j];
    }
    for (int i = 0; i < height; i++) {
        dst[i * width + 0]           = src[i * width + 0];
        dst[i * width + (width-1)]   = src[i * width + (width-1)];
    }
}
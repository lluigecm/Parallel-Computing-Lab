#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "gaussian.h"


// ─── Alocação ────────────────────────────────────────────────────────────────

float* alloc_img(int height, int width) {
    float* img = (float*)malloc(height * width * sizeof(float));
    if (img == NULL) {
        fprintf(stderr, "Erro: falha ao alocar imagem\n");
        exit(1);
    }
    return img;
}

// ─── Geração do kernel gaussiano ─────────────────────────────────────────────
// Calcula os pesos usando G(x,y) = exp(-(x²+y²) / (2*sigma²))
// e normaliza para que a soma dos pesos seja 1.

void generate_kernel(float kernel[KERNEL_SIZE][KERNEL_SIZE], float sigma) {
    int half = KERNEL_SIZE / 2;
    float sum = 0.0f;

    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            int x = i - half;
            int y = j - half;
            kernel[i][j] = expf(-(float)(x*x + y*y) / (2.0f * sigma * sigma));
            sum += kernel[i][j];
        }
    }

    // Normalização: divide cada peso pelo total
    for (int i = 0; i < KERNEL_SIZE; i++)
        for (int j = 0; j < KERNEL_SIZE; j++)
            kernel[i][j] /= sum;
}


// ─── Carreggamento e salvamento de imagem ─────────────────────────────────────────────

float* load_img(const char* filename, int* width, int* height) {
    int channels;
    // Força leitura em escala de cinza)
    unsigned char* data = stbi_load(filename, width, height, &channels, 1);
    if (data == NULL) {
        fprintf(stderr, "Erro ao carregar imagem: %s\n", filename);
        exit(1);
    }

    float* img = alloc_img(*height, *width);
    for (int i = 0; i < (*height) * (*width); i++) {
        img[i] = (float)data[i];
    }

    stbi_image_free(data);
    return img;
}

void save_img(const char* filename, float* img, int height, int width) {
    unsigned char* data = (unsigned char*)malloc(height * width);
    if (data == NULL) {
        fprintf(stderr, "Erro ao alocar buffer de saída\n");
        exit(1);
    }

    for (int i = 0; i < height * width; i++) {
        float val = img[i];
        if (val < 0.0f)   val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        data[i] = (unsigned char)val;
    }

    stbi_write_png(filename, width, height, 1, data, width);
    free(data);
}


// ─── Filtro gaussiano — Serial ────────────────────────────────────────────────

void gaussian_blur_serial(float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width) {

    // Copia bordas sem filtrar
    for (int j = 0; j < width; j++) {
        dst[0 * width + j]           = src[0 * width + j];
        dst[(height-1) * width + j]  = src[(height-1) * width + j];
    }
    for (int i = 0; i < height; i++) {
        dst[i * width + 0]           = src[i * width + 0];
        dst[i * width + (width-1)]   = src[i * width + (width-1)];
    }

    // Aplica filtro nos pixels internos
    // Como o kernel já está normalizado, a soma == 1 e não precisa dividir
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0f;
            for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                    sum += src[(i+k) * width + (j+l)] * kernel[k+1][l+1];
            dst[i * width + j] = sum;
        }
    }
}


// ─── Filtro gaussiano — OpenMP ────────────────────────────────────────────────

void gaussian_blur_openmp(float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width) {

    // Copia bordas sem filtrar
    for (int j = 0; j < width; j++) {
        dst[0 * width + j]           = src[0 * width + j];
        dst[(height-1) * width + j]  = src[(height-1) * width + j];
    }
    for (int i = 0; i < height; i++) {
        dst[i * width + 0]           = src[i * width + 0];
        dst[i * width + (width-1)]   = src[i * width + (width-1)];
    }

    // Paraleliza o loop externo entre as threads
    #pragma omp parallel for
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0f;
            for (int k = -1; k <= 1; k++)
                for (int l = -1; l <= 1; l++)
                    sum += src[(i+k) * width + (j+l)] * kernel[k+1][l+1];
            dst[i * width + j] = sum;
        }
    }
}


// ─── Filtro gaussiano — Tiling ────────────────────────────────────────────────

void gaussian_blur_tiling(float* src, float* dst, float kernel[KERNEL_SIZE][KERNEL_SIZE], int height, int width) {
    int TILE = 64;

    // Copia bordas sem filtrar
    for (int j = 0; j < width; j++) {
        dst[0 * width + j]           = src[0 * width + j];
        dst[(height-1) * width + j]  = src[(height-1) * width + j];
    }
    for (int i = 0; i < height; i++) {
        dst[i * width + 0]           = src[i * width + 0];
        dst[i * width + (width-1)]   = src[i * width + (width-1)];
    }

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ti = 1; ti < height - 1; ti += TILE) {
        for (int tj = 1; tj < width - 1; tj += TILE) {

            int i_end = ti + TILE < height - 1 ? ti + TILE : height - 1;
            int j_end = tj + TILE < width  - 1 ? tj + TILE : width  - 1;

            for (int i = ti; i < i_end; i++) {
                for (int j = tj; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int k = -1; k <= 1; k++)
                        for (int l = -1; l <= 1; l++)
                            sum += src[(i+k) * width + (j+l)] * kernel[k+1][l+1];
                    dst[i * width + j] = sum;
                }
            }
        }
    }
}

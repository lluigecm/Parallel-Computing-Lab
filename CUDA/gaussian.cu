#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gaussian.cuh"

// ─── Alocacao ─────────────────────────────────────────────────────────────────

float* alloc_img(int height, int width) {
    float* img = (float*)malloc(height * width * sizeof(float));
    if (img == NULL) {
        fprintf(stderr, "Erro: falha ao alocar imagem\n");
        exit(1);
    }
    return img;
}

// ─── Kernel gaussiano ─────────────────────────────────────────────────────────
// kernel_size vem como argumento e o kernel e retornado como array 1D

float* generate_kernel(int kernel_size, float sigma) {
    int half = kernel_size / 2;
    float* k = (float*)malloc(kernel_size * kernel_size * sizeof(float));
    if (!k) { fprintf(stderr, "Erro: malloc kernel\n"); exit(1); }

    float sum = 0.0f;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            int x = i - half, y = j - half;
            k[i * kernel_size + j] = expf(-(float)(x*x + y*y) / (2.0f * sigma * sigma));
            sum += k[i * kernel_size + j];
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; i++)
        k[i] /= sum;

    return k;
}

// ─── I/O ──────────────────────────────────────────────────────────────────────

float* load_img(const char* filename, int* width, int* height) {
    int channels;
    unsigned char* data = stbi_load(filename, width, height, &channels, 1);
    if (!data) { fprintf(stderr, "Erro ao carregar: %s\n", filename); exit(1); }

    float* img = alloc_img(*height, *width);
    for (int i = 0; i < (*height) * (*width); i++)
        img[i] = (float)data[i];

    stbi_image_free(data);
    return img;
}

void save_img(const char* filename, float* img, int height, int width) {
    unsigned char* data = (unsigned char*)malloc(height * width);
    if (!data) { fprintf(stderr, "Erro: malloc save\n"); exit(1); }

    for (int i = 0; i < height * width; i++) {
        float v = img[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        data[i] = (unsigned char)v;
    }
    stbi_write_png(filename, width, height, 1, data, width);
    free(data);
}

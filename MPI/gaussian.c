#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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


// ─── Kernel gaussiano ─────────────────────────────────────────────────────────
// Diferente da versao OpenMP, aqui o kernel_size vem como argumento
// e o kernel e retornado como array 1D alocado dinamicamente

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


// ─── Filtro local (MPI) ───────────────────────────────────────────────────────
//
// O buffer tem "half" linhas de halo em cima e embaixo das linhas reais.
// Bordas absolutas da imagem (primeiras/ultimas `half` linhas e colunas)
// sao copiadas sem filtrar, igual ao serial.

void gaussian_blur_local(float* src, float* dst, float* kernel, int kernel_size,
                          int local_rows, int width, int first, int last) {
    int half = kernel_size / 2;

    // Copia colunas de borda
    for (int i = half; i < half + local_rows; i++) {
        for (int j = 0; j < half; j++) {
            dst[i * width + j]           = src[i * width + j];
            dst[i * width + (width-1-j)] = src[i * width + (width-1-j)];
        }
    }

    // Copia linhas de borda superior (so o primeiro processo)
    if (first) {
        for (int i = half; i < half + half; i++)
            for (int j = 0; j < width; j++)
                dst[i * width + j] = src[i * width + j];
    }

    // Copia linhas de borda inferior (so o ultimo processo)
    if (last) {
        for (int i = half + local_rows - half; i < half + local_rows; i++)
            for (int j = 0; j < width; j++)
                dst[i * width + j] = src[i * width + j];
    }

    int i_start = half + (first ? half : 0);
    int i_end   = half + local_rows - (last ? half : 0);

    for (int i = i_start; i < i_end; i++) {
        for (int j = half; j < width - half; j++) {
            float sum = 0.0f;
            for (int ki = -half; ki <= half; ki++)
                for (int kj = -half; kj <= half; kj++)
                    sum += src[(i + ki) * width + (j + kj)]
                         * kernel[(ki + half) * kernel_size + (kj + half)];
            dst[i * width + j] = sum;
        }
    }
}

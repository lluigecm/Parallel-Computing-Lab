#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "gaussian.cuh"

// Uso: ./cuda <imagem.png> [iteracoes] [kernel_size] [sigma] [repeticoes] [modo]
// defaults: iteracoes=1000, kernel_size=3, sigma=1.0, repeticoes=5, modo=shared
// modo: "global" ou "shared"

#define MAX_KERNEL 11   // suporta kernels de ate 11x11
#define TILE 16         // bloco 16x16 = 256 threads

// Kernel gaussiano fica na memoria constante (imutavel, acesso broadcast)
__constant__ float d_kernel[MAX_KERNEL * MAX_KERNEL];


// ─── Filtro em memoria global ─────────────────────────────────────────────────
// Uma thread por pixel, lendo os vizinhos direto da memoria global.

__global__ void blur_global(const float* src, float* dst,
                            int height, int width, int kernel_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= height || j >= width) return;

    int half = kernel_size / 2;

    // Bordas copiadas sem filtrar, igual ao serial
    if (i < half || i >= height - half || j < half || j >= width - half) {
        dst[i * width + j] = src[i * width + j];
        return;
    }

    float sum = 0.0f;
    for (int ki = -half; ki <= half; ki++)
        for (int kj = -half; kj <= half; kj++)
            sum += src[(i + ki) * width + (j + kj)]
                 * d_kernel[(ki + half) * kernel_size + (kj + half)];

    dst[i * width + j] = sum;
}


// ─── Filtro com memoria compartilhada (tiling) ────────────────────────────────
// Cada bloco carrega um tile TILE x TILE mais o halo na shared memory.
// O halo sao as "half" linhas/colunas de vizinhos que a convolucao precisa.

__global__ void blur_shared(const float* src, float* dst,
                            int height, int width, int kernel_size) {
    int half = kernel_size / 2;

    __shared__ float tile[TILE + 2 * (MAX_KERNEL / 2)][TILE + 2 * (MAX_KERNEL / 2)];
    int sdim = TILE + 2 * half;

    int tx = threadIdx.x, ty = threadIdx.y;
    int j = blockIdx.x * TILE + tx;
    int i = blockIdx.y * TILE + ty;

    // Canto superior-esquerdo do tile na imagem (ja inclui o halo)
    int base_i = blockIdx.y * TILE - half;
    int base_j = blockIdx.x * TILE - half;

    // Carregamento cooperativo: preenche o tile inteiro. Fora da imagem vira 0
    // (esses zeros nunca entram numa saida real porque as bordas sao copiadas)
    for (int ly = ty; ly < sdim; ly += TILE) {
        for (int lx = tx; lx < sdim; lx += TILE) {
            int gy = base_i + ly;
            int gx = base_j + lx;
            float v = 0.0f;
            if (gy >= 0 && gy < height && gx >= 0 && gx < width)
                v = src[gy * width + gx];
            tile[ly][lx] = v;
        }
    }

    __syncthreads();

    if (i >= height || j >= width) return;

    // Bordas copiadas sem filtrar
    if (i < half || i >= height - half || j < half || j >= width - half) {
        dst[i * width + j] = src[i * width + j];
        return;
    }

    // Convolucao lendo so da shared memory
    float sum = 0.0f;
    for (int ki = -half; ki <= half; ki++)
        for (int kj = -half; kj <= half; kj++)
            sum += tile[ty + half + ki][tx + half + kj]
                 * d_kernel[(ki + half) * kernel_size + (kj + half)];

    dst[i * width + j] = sum;
}


int main(int argc, char* argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Uso: %s <imagem.png> [iteracoes] [kernel_size] [sigma] [repeticoes] [modo]\n", argv[0]);
        fprintf(stderr, "Exemplo: %s ../image_4096x4096.png 1000 3 1.0 5 shared\n", argv[0]);
        return 1;
    }

    const char* filename    = argv[1];
    int         iterations  = (argc > 2) ? atoi(argv[2])        : 1000;
    int         kernel_size = (argc > 3) ? atoi(argv[3])        : 3;
    float       sigma       = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
    int         repeticoes  = (argc > 5) ? atoi(argv[5])        : 5;
    const char* modo        = (argc > 6) ? argv[6]              : "shared";

    if (kernel_size < 3 || kernel_size % 2 == 0) {
        fprintf(stderr, "Erro: kernel_size deve ser impar e >= 3\n");
        return 1;
    }
    if (kernel_size > MAX_KERNEL) {
        fprintf(stderr, "Erro: kernel_size maximo e %d\n", MAX_KERNEL);
        return 1;
    }
    int usar_shared = (strcmp(modo, "global") != 0);

    // ─── Carrega imagem e gera kernel ─────────────────────────────────────────
    int width = 0, height = 0;
    float* h_img    = load_img(filename, &width, &height);
    float* h_kernel = generate_kernel(kernel_size, sigma);
    size_t img_bytes = (size_t)height * width * sizeof(float);

    // Copia o kernel para a memoria constante (uma vez so)
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size * kernel_size * sizeof(float));

    // ─── Aloca buffers no device ──────────────────────────────────────────────
    float *d_src, *d_dst;
    cudaMalloc(&d_src, img_bytes);
    cudaMalloc(&d_dst, img_bytes);

    // ─── Configuracao da grade ────────────────────────────────────────────────
    dim3 bloco(TILE, TILE);
    dim3 grade((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);

    // ─── Cabecalho ────────────────────────────────────────────────────────────
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("==============================================\n");
    printf("Modo:         CUDA (%s)\n", usar_shared ? "shared" : "global");
    printf("GPU:          %s\n", prop.name);
    printf("Arquivo:      %s (%dx%d)\n", filename, width, height);
    printf("Iteracoes:    %d\n", iterations);
    printf("Repeticoes:   %d\n", repeticoes);
    printf("Sigma:        %.2f\n", sigma);
    printf("Kernel:       %dx%d\n", kernel_size, kernel_size);
    printf("Bloco:        %dx%d\n", TILE, TILE);
    printf("==============================================\n\n");

    // ─── Benchmark ────────────────────────────────────────────────────────────
    // Tempo medido com eventos CUDA, so o laco de iteracoes (sem as copias)
    cudaEvent_t inicio, fim;
    cudaEventCreate(&inicio);
    cudaEventCreate(&fim);

    double soma = 0.0;

    for (int rep = 0; rep < repeticoes; rep++) {
        cudaMemcpy(d_src, h_img, img_bytes, cudaMemcpyHostToDevice);

        cudaEventRecord(inicio);

        float *src = d_src, *dst = d_dst;
        for (int it = 0; it < iterations; it++) {
            if (usar_shared)
                blur_shared<<<grade, bloco>>>(src, dst, height, width, kernel_size);
            else
                blur_global<<<grade, bloco>>>(src, dst, height, width, kernel_size);

            float* tmp = src; src = dst; dst = tmp;
        }

        cudaEventRecord(fim);
        cudaEventSynchronize(fim);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, inicio, fim);
        double tempo = ms / 1000.0;
        printf("  Repeticao %d: %.6f s\n", rep + 1, tempo);
        soma += tempo;

        // Na ultima repeticao, src aponta para o resultado final
        if (rep == repeticoes - 1) {
            float* h_out = alloc_img(height, width);
            cudaMemcpy(h_out, src, img_bytes, cudaMemcpyDeviceToHost);
            char output[64];
            snprintf(output, sizeof(output), "output_cuda_%s.png", usar_shared ? "shared" : "global");
            save_img(output, h_out, height, width);
            printf("\nResultado salvo em: %s\n", output);
            free(h_out);
        }
    }

    printf("Tempo medio: %.6f s\n", soma / repeticoes);

    // ─── Liberacao de memoria ─────────────────────────────────────────────────
    cudaEventDestroy(inicio);
    cudaEventDestroy(fim);
    cudaFree(d_src);
    cudaFree(d_dst);
    free(h_img);
    free(h_kernel);

    return 0;
}

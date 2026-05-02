#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "gaussian.h"

// Uso: ./serial <arquivo.png> <iteracoes> <sigma> <repeticoes>

int main(int argc, char* argv[]) {

    if (argc != 5) {
        fprintf(stderr, "Uso: %s <arquivo.png> <iteracoes> <sigma> <repeticoes>\n", argv[0]);
        fprintf(stderr, "Exemplo: %s image_4096x4096.png 100 1.0 5\n", argv[0]);
        return 1;
    }

    const char* filename   = argv[1];
    int         iterations = atoi(argv[2]);
    float       sigma      = (float)atof(argv[3]);
    int         repeticoes = atoi(argv[4]);

    // ─── Carrega imagem e gera kernel ──────────────────────────────────────
    int height, width;
    float* original = load_img(filename, &width, &height);

    float kernel[KERNEL_SIZE][KERNEL_SIZE];
    generate_kernel(kernel, sigma);

    float* src = alloc_img(height, width);
    float* dst = alloc_img(height, width);

    // ─── Cabeçalho ─────────────────────────────────────────────────────────
    printf("==============================================\n");
    printf("Modo:         serial\n");
    printf("Arquivo:      %s (%dx%d)\n", filename, width, height);
    printf("Iterações:    %d\n", iterations);
    printf("Repetições:   %d\n", repeticoes);
    printf("Sigma:        %.2f\n", sigma);
    printf("==============================================\n\n");

    // ─── Benchmark ─────────────────────────────────────────────────────────
    double soma = 0.0;

    for (int r = 0; r < repeticoes; r++) {
        memcpy(src, original, height * width * sizeof(float));

        double t_inicio = omp_get_wtime();
        for (int i = 0; i < iterations; i++) {
            gaussian_blur_serial(src, dst, kernel, height, width);
            float* tmp = src; src = dst; dst = tmp;
        }
        double tempo = omp_get_wtime() - t_inicio;

        printf("  Repetição %d: %.6f s\n", r + 1, tempo);
        soma += tempo;
    }

    double media = soma / repeticoes;
    printf("\nTempo médio: %.6f s\n", media);

    // ─── Salva resultado ───────────────────────────────────────────────────
    save_img("output_serial.png", src, height, width);
    printf("Resultado salvo em: output_serial.png\n");

    free(original);
    free(src);
    free(dst);

    return 0;
}
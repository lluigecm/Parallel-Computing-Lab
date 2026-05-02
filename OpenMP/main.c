#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "gaussian.h"


int main(int argc, char* argv[]) {
    
    if (argc != 4) {
        fprintf(stderr, "Uso: %s <altura> <largura> <iteracoes>\n", argv[0]);
        return 1;
    }
    
    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    
    // Aloca e inicializa a imagem
    float* src = alloc_img(height, width);
    float* dst = alloc_img(height, width);
    
    srand(23); // Semente para geração de números aleatórios
    
    init_img(src, height, width);
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int i = 0; i < iterations; i++) {
        gaussian_blur(src, dst, height, width);

        // Troca os ponteiros para a próxima iteração
        float* temp = src;
        src = dst;
        dst = temp;
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Tempo gasto: %.6f segundos\n", elapsed_time);

    free(src);
    free(dst);

    return 0;
    
}
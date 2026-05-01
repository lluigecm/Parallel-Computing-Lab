#include <stdio.h>
#include "gaussian.h"



float* alloc_img(int height, int width) {
    float* img = (float*)malloc(height * width * sizeof(float));

    if (img == NULL) {
        fprintf(stderr, "Erro: falha ao alocar imagem\n");
        exit(1);
    }
    return img;
}
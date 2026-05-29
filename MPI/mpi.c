#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "gaussian.h"

// Uso: mpirun -np N ./mpi <imagem.png> <iteracoes> <kernel_size> <sigma> <repeticoes>
// kernel_size deve ser impar (ex: 3, 5, 7)

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 6) {
        if (rank == 0)
            fprintf(stderr, "Uso: mpirun -np N %s <imagem.png> <iteracoes> <kernel_size> <sigma> <repeticoes>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char* filename    = argv[1];
    int         iterations  = atoi(argv[2]);
    int         kernel_size = atoi(argv[3]);
    float       sigma       = (float)atof(argv[4]);
    int         repeticoes  = atoi(argv[5]);

    if (kernel_size < 3 || kernel_size % 2 == 0) {
        if (rank == 0) fprintf(stderr, "Erro: kernel_size deve ser impar e >= 3\n");
        MPI_Finalize();
        return 1;
    }

    int half = kernel_size / 2;

    // Cada processo gera o kernel localmente (mesmo resultado, evita broadcast)
    float* kernel = generate_kernel(kernel_size, sigma);

    // ─── Rank 0 carrega a imagem ──────────────────────────────────────────────
    int width = 0, height = 0;
    float* full_img = NULL;

    if (rank == 0)
        full_img = load_img(filename, &width, &height);

    MPI_Bcast(&width,  1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ─── Distribuicao de linhas ───────────────────────────────────────────────
    int base = height / size;
    int extra = height % size;

    int* local_rows = (int*)malloc(size * sizeof(int));
    int* counts     = (int*)malloc(size * sizeof(int));
    int* displs     = (int*)malloc(size * sizeof(int));

    int offset = 0;
    for (int r = 0; r < size; r++) {
        local_rows[r] = base + (r < extra ? 1 : 0);
        counts[r]     = local_rows[r] * width;
        displs[r]     = offset;
        offset       += counts[r];
    }

    int my_rows = local_rows[rank];

    // Buffer com espaco para os halos (half linhas em cima e embaixo)
    int buf_rows = my_rows + 2 * half;
    float* src = (float*)calloc(buf_rows * width, sizeof(float));
    float* dst = (float*)calloc(buf_rows * width, sizeof(float));
    float* orig = (float*)malloc(my_rows * width * sizeof(float));

    // ─── Scatter ──────────────────────────────────────────────────────────────

    MPI_Scatterv(full_img, counts, displs, MPI_FLOAT,
                 src + half * width, my_rows * width, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    memcpy(orig, src + half * width, my_rows * width * sizeof(float));

    if (rank == 0) { free(full_img); full_img = NULL; }

    // MPI_PROC_NULL nos extremos faz o MPI ignorar essas comunicações
    int top    = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int bottom = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    int first  = (rank == 0);
    int last   = (rank == size - 1);

    // ─── Cabecalho ────────────────────────────────────────────────────────────
    if (rank == 0) {
        printf("==============================================\n");
        printf("Modo:         MPI\n");
        printf("Arquivo:      %s (%dx%d)\n", filename, width, height);
        printf("Iteracoes:    %d\n", iterations);
        printf("Repeticoes:   %d\n", repeticoes);
        printf("Sigma:        %.2f\n", sigma);
        printf("Kernel:       %dx%d\n", kernel_size, kernel_size);
        printf("Processos:    %d\n", size);
        printf("==============================================\n\n");
    }

    // ─── Benchmark ────────────────────────────────────────────────────────────
    MPI_Status status;
    double soma = 0.0;

    for (int rep = 0; rep < repeticoes; rep++) {
        memcpy(src + half * width, orig, my_rows * width * sizeof(float));

        MPI_Barrier(MPI_COMM_WORLD);
        double t_inicio = MPI_Wtime();

        for (int it = 0; it < iterations; it++) {

            // Troca de halos: cada processo envia suas bordas reais para os vizinhos
            // e recebe as bordas deles para preencher os halos locais.
            // Duas chamadas com tags diferentes para evitar deadlock.

            // Envia primeiras linhas reais para cima, recebe halo inferior de baixo
            MPI_Sendrecv(src + half * width,              half * width, MPI_FLOAT, top,    0,
                         src + (half + my_rows) * width,  half * width, MPI_FLOAT, bottom, 0,
                         MPI_COMM_WORLD, &status);

            // Envia ultimas linhas reais para baixo, recebe halo superior de cima
            MPI_Sendrecv(src + my_rows * width,  half * width, MPI_FLOAT, bottom, 1,
                         src,                    half * width, MPI_FLOAT, top,    1,
                         MPI_COMM_WORLD, &status);

            gaussian_blur_local(src, dst, kernel, kernel_size, my_rows, width, first, last);

            float* tmp = src; src = dst; dst = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double tempo = MPI_Wtime() - t_inicio;

        if (rank == 0) {
            printf("  Repeticao %d: %.6f s\n", rep + 1, tempo);
            soma += tempo;
        }
    }

    if (rank == 0)
        printf("\nTempo medio: %.6f s\n", soma / repeticoes);

    // ─── Gather e salvar ──────────────────────────────────────────────────────
    float* result = NULL;
    if (rank == 0)
        result = (float*)malloc(height * width * sizeof(float));

    MPI_Gatherv(src + half * width, my_rows * width, MPI_FLOAT,
                result, counts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        char output[64];
        snprintf(output, sizeof(output), "output_mpi_%dprocs.png", size);
        save_img(output, result, height, width);
        printf("Resultado salvo em: %s\n", output);
        free(result);
    }

    // ─── LIberação de Memória ──────────────────────────────────────────────────────────────
    free(src);
    free(dst);
    free(orig);
    free(kernel);
    free(local_rows);
    free(counts);
    free(displs);

    MPI_Finalize();
    return 0;
}

# Parallel-Computing-Lab

Projeto desenvolvido para a disciplina **DEC107 — Processamento Paralelo**.  
Benchmark de suavização Gaussiana iterativa sobre imagens 2D, comparando implementações serial, paralela com OpenMP e distribuída com MPI.

**Autores:** Lucas Luige Costa Miranda, Henrique Cerqueira Fadigas

---

## Estrutura do repositório

```
Parallel-Computing-Lab/
├── OpenMP/
│   ├── gaussian.h          # protótipos e defines
│   ├── gaussian.c          # implementação das funções (serial, openmp, tiling)
│   ├── serial.c            # main do benchmark serial
│   ├── openmp.c            # main do benchmark OpenMP
│   ├── tiling.c            # main do benchmark OpenMP com tiling (extra)
│   ├── Makefile            # compila os três binários
│   ├── resultados.txt      # tempos de execução (serial e OpenMP)
│   ├── testes.txt          # comandos exatos utilizados nos testes
│   └── Relatorio/
│       ├── relatorio.pdf       # relatório (feito em LaTeX)
│       ├── gerar_graficos.py   # gera os gráficos (requer matplotlib)
│       └── figuras/            # gráficos gerados (PDF)
└── MPI/
    ├── gaussian.h          # protótipos e defines
    ├── gaussian.c          # implementação das funções
    ├── mpi.c               # main do benchmark MPI
    ├── Makefile            # compila o binário mpi
    ├── analise.txt         # análise comparativa Serial vs OpenMP vs MPI
    ├── resultados.txt      # tempos de execução (serial, OpenMP e MPI)
    └── testes.txt          # comandos exatos utilizados nos testes
```

---

## Dependências

- GCC com suporte a OpenMP (`-fopenmp`)
- MPI (ex: OpenMPI) com `mpicc` e `mpirun`
- Python 3 com `matplotlib` e `numpy` (para gerar as imagens de teste e os gráficos)

```bash
# Instalar dependências Python
pip install matplotlib numpy

# Instalar OpenMPI (Ubuntu/Debian)
sudo apt install libopenmpi-dev openmpi-bin
```

---

## Como usar

### 1. Gerar as imagens de teste

As imagens de entrada precisam ser geradas uma única vez a partir de `base.png`:

```bash
python3 gerarImagens.py
```

Isso cria `image_512x512.png`, `image_1024x1024.png`, `image_2048x2048.png` e `image_4096x4096.png` na raiz do repositório.

### 2. Compilar os binários

**OpenMP (serial, openmp, tiling):**
```bash
cd OpenMP/
make
```

**MPI:**
```bash
cd MPI/
make
```

Para remover binários e imagens de saída: `make clean` em cada diretório.

### 3. Executar os benchmarks

**Serial e OpenMP** (executados a partir de `OpenMP/`):

```
./serial  <imagem.png> <iterações> <sigma> <repetições>
OMP_NUM_THREADS=N ./openmp  <imagem.png> <iterações> <sigma> <repetições>
OMP_NUM_THREADS=N ./tiling  <imagem.png> <iterações> <sigma> <repetições>
```

**MPI** (executado a partir de `MPI/`):

```
mpirun -np N ./mpi <imagem.png> <iterações> <kernel_size> <sigma> <repetições>
```

> Para usar mais processos do que cores físicos (ex: 16 em máquina com 8 cores), adicione a flag `--use-hwthread-cpus`.

| Parâmetro     | Descrição                                                   |
|---------------|-------------------------------------------------------------|
| `imagem.png`  | Arquivo de entrada (escala de cinza)                        |
| `iterações`   | Número de vezes que o filtro é aplicado                     |
| `kernel_size` | Tamanho do kernel Gaussiano — deve ser ímpar e ≥ 3 (MPI)   |
| `sigma`       | Desvio padrão do kernel Gaussiano                           |
| `repetições`  | Número de repetições para calcular a média                  |

**Exemplo — parâmetros usados nos testes do projeto:**

```bash
# Serial
./serial ../image_4096x4096.png 1000 1.0 5

# OpenMP com 4 threads
OMP_NUM_THREADS=4 ./openmp ../image_4096x4096.png 1000 1.0 5

# MPI com 4 processos
mpirun -np 4 ./mpi ../image_4096x4096.png 1000 3 1.0 5

# MPI com 16 processos (usando hyperthreading)
mpirun --use-hwthread-cpus -np 16 ./mpi ../image_4096x4096.png 1000 3 1.0 5
```

A saída é salva automaticamente em `output_serial.png`, `output_openmp_Nthreads.png` ou `output_mpi_Nprocs.png`.

---

## Relatório e gráficos

O relatório detalha a implementação OpenMP, análise de desempenho e conclusões.  
O script `gerar_graficos.py` gera os gráficos de tempo, speedup e eficiência a partir dos arquivos de saída dos benchmarks.  
Por escolha, o código LaTeX do relatório não está incluído no repositório, pois necessitaria de dependências adicionais para compilar (como `pdflatex` e pacotes LaTeX). O relatório finalizado está disponível em PDF.

---

## Testes

Cada diretório (`OpenMP/` e `MPI/`) contém um arquivo `testes.txt` com o comando exato usado para cada teste, incluindo os parâmetros e o número de threads/processos. Ele serve como referência para os dados de desempenho apresentados no relatório.

## Resultados

Cada diretório contém um arquivo `resultados.txt` com os tempos de execução brutos de cada repetição e o tempo médio calculado. O arquivo em `MPI/resultados.txt` consolida os resultados das três implementações (serial, OpenMP e MPI) e é a base para a análise comparativa em `MPI/analise.txt`.

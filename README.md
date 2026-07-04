# Parallel-Computing-Lab

Projeto desenvolvido para a disciplina **DEC107 — Processamento Paralelo**.  
Benchmark de suavização Gaussiana iterativa sobre imagens 2D, comparando implementações serial, paralela com OpenMP (Etapa 1), distribuída com MPI (Etapa 2) e massivamente paralela em GPU com CUDA (Etapa 3).

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
├── MPI/
│   ├── gaussian.h          # protótipos e defines
│   ├── gaussian.c          # implementação das funções
│   ├── mpi.c               # main do benchmark MPI
│   ├── Makefile            # compila o binário mpi
│   ├── analise.txt         # análise comparativa Serial vs OpenMP vs MPI
│   ├── resultados.txt      # tempos de execução (serial, OpenMP e MPI)
│   ├── testes.txt          # comandos exatos utilizados nos testes
│   └── Relatorio/
│       ├── relatorio.pdf       # relatório (feito em LaTeX)
│       ├── gerar_graficos.py   # gera os gráficos (requer matplotlib)
│       └── figuras/            # gráficos gerados (PDF)
├── CUDA/
│   ├── gaussian.cuh        # protótipos das funções de host
│   ├── gaussian.cu         # funções de host (I/O via stb, geração do kernel)
│   ├── cuda.cu             # kernels de device (global e shared) + benchmark
│   ├── Makefile            # compila com nvcc (-arch=sm_75)
│   ├── testes.txt          # comandos exatos utilizados nos testes
│   └── Relatorio/
│       ├── relatorio.pdf       # relatório da Etapa 3 (feito em LaTeX)
│       ├── gerar_graficos.py   # gera os gráficos (requer matplotlib)
│       └── figuras/            # gráficos gerados (PDF)
├── resultados_i5-11300h.txt   # tempos brutos das 4 implementações no mesmo notebook
```

---

## Dependências

- GCC com suporte a OpenMP (`-fopenmp`)
- MPI (ex: OpenMPI) com `mpicc` e `mpirun`
- CUDA Toolkit com `nvcc` e driver NVIDIA (para a Etapa 3) — GPU testada: NVIDIA GTX 1650, arquitetura Turing (`sm_75`)
- Python 3 com `matplotlib` e `numpy` (para gerar as imagens de teste e os gráficos)

```bash
# Instalar dependências Python
pip install matplotlib numpy

# Instalar OpenMPI (Ubuntu/Debian)
sudo apt install libopenmpi-dev openmpi-bin
```

> **Nota de toolchain (CUDA):** em Ubuntu 26.04 (glibc 2.43), o CUDA Toolkit 12.6 não compila por conflito nas declarações de `rsqrt`/`cospi`/`sinpi`. Foi usado o **CUDA Toolkit 13.1** com a flag `-Xcompiler -U_GNU_SOURCE` (já configurada no `CUDA/Makefile`). Se a GPU não for uma GTX 1650, ajuste `-arch=sm_XX` no `CUDA/Makefile`.

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

**CUDA:**
```bash
cd CUDA/
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

**CUDA** (executado a partir de `CUDA/`):

```
./cuda <imagem.png> [iterações] [kernel_size] [sigma] [repetições] [modo]
```

> `modo` é `global` ou `shared`. Todos os argumentos além da imagem são opcionais; os defaults são `1000 3 1.0 5 shared`. O tempo é medido com eventos CUDA (só o laço de kernel, sem as cópias host↔device). A saída é salva em `output_cuda_global.png` ou `output_cuda_shared.png`.

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

# MPI com 8 processos (usando hyperthreading em CPU de 4 núcleos físicos)
mpirun --use-hwthread-cpus -np 8 ./mpi ../image_4096x4096.png 1000 3 1.0 5

# CUDA, versão memória global
./cuda ../image_4096x4096.png 1000 3 1.0 5 global

# CUDA, versão memória compartilhada (shared/tiling)
./cuda ../image_4096x4096.png 1000 3 1.0 5 shared
```

A saída é salva automaticamente em `output_serial.png`, `output_openmp_Nthreads.png`, `output_mpi_Nprocs.png` ou `output_cuda_{global,shared}.png`.

---

## Relatório e gráficos

Cada etapa possui seu próprio relatório em PDF em `OpenMP/Relatorio/`, `MPI/Relatorio/` e `CUDA/Relatorio/`.
O script `gerar_graficos.py` presente em cada diretório gera os gráficos de tempo, speedup e eficiência (requer `matplotlib`).
Por escolha, o código-fonte LaTeX não está incluído no repositório. Os relatórios finalizados estão disponíveis em PDF.

---

## Testes

Cada diretório (`OpenMP/`, `MPI/` e `CUDA/`) contém um arquivo `testes.txt` com o comando exato usado para cada teste, incluindo os parâmetros e o número de threads/processos. Ele serve como referência para os dados de desempenho apresentados no relatório.

## Resultados

Cada diretório contém um arquivo `resultados.txt` com os tempos de execução brutos de cada repetição e o tempo médio calculado. O arquivo em `MPI/resultados.txt` consolida os resultados das três primeiras implementações (serial, OpenMP e MPI) executadas no hardware das Etapas 1 e 2 (AMD Ryzen 7 5700G).

Na Etapa 3, as **quatro** implementações (serial, OpenMP, MPI e CUDA) foram reexecutadas no **mesmo notebook** (Intel Core i5-11300H + GTX 1650), para uma comparação consistente no mesmo hardware. Os tempos brutos estão em `resultados_i5-11300h.txt`.

---

## Nota de Transparência sobre o Uso de IA

Declaro que este projeto contou com o auxílio da ferramenta de IA **Claude (Anthropic)** exclusivamente para as tarefas de depuração e estruturação das implementações (OpenMP, MPI e os kernels CUDA), resolução do conflito de toolchain entre o CUDA e o glibc do Ubuntu 26.04, discussão e interpretação dos resultados de desempenho, organização dos arquivos de análise e estruturação dos relatórios em LaTeX. Como autores, atestamos que revisamos, testamos e validamos criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos relatórios de desempenho e integridade acadêmica do material entregue.


Lucas Luige Costa Miranda — 04 de julho de 2026
Henrique Cerqueira Fadigas — 04 de julho de 2026
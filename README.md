# Parallel-Computing-Lab

Projeto desenvolvido para a disciplina **DEC107 — Processamento Paralelo**.  
Benchmark de suavização Gaussiana iterativa sobre imagens 2D, comparando implementações serial e paralela com OpenMP.

**Autores:** Lucas Luige Costa Miranda, Henrique Cerqueira Fadigas

---

## Estrutura do repositório

```
Parallel-Computing-Lab/
└── OpenMP/
    ├── gaussian.h          # protótipos e defines
    ├── gaussian.c          # implementação das funções (serial, openmp, tiling)
    ├── serial.c            # main do benchmark serial
    ├── openmp.c            # main do benchmark OpenMP
    ├── tiling.c            # main do benchmark OpenMP com tiling (extra)
    ├── Makefile            # compila os três binários
    ├── gerarImagens.py     # gera as imagens de teste a partir de base.png
    ├── base.png            # imagem base para geração dos testes
    └── Relatorio/
        ├── relatorio.pdf       # relatório (feito em LaTeX)
        ├── gerar_graficos.py   # gera os gráficos (requer matplotlib)
        └── figuras/            # gráficos gerados (PDF)
```

---

## Dependências

- GCC com suporte a OpenMP (`-fopenmp`)
- Python 3 com `matplotlib` e `numpy` (para gerar as imagens de teste e os gráficos)

```bash
# Instalar dependências Python
pip install matplotlib numpy

```

---

## Como usar

### 1. Gerar as imagens de teste

As imagens de entrada precisam ser geradas uma única vez a partir de `base.png`:

```bash
cd OpenMP/
python3 gerarImagens.py
```

Isso cria `image_512x512.png`, `image_1024x1024.png`, `image_2048x2048.png` e `image_4096x4096.png`.

### 2. Compilar os binários

```bash
cd OpenMP/
make
```

Gera três binários: `serial`, `openmp` e `tiling`.  
Para remover os binários e imagens de saída: `make clean`

### 3. Executar os benchmarks

Todos os binários seguem a mesma assinatura:

```
./serial  <imagem.png> <iterações> <sigma> <repetições>
OMP_NUM_THREADS=N ./openmp  <imagem.png> <iterações> <sigma> <repetições>
OMP_NUM_THREADS=N ./tiling  <imagem.png> <iterações> <sigma> <repetições>
```

| Parâmetro     | Descrição                                      |
|---------------|------------------------------------------------|
| `imagem.png`  | Arquivo de entrada (escala de cinza)           |
| `iterações`   | Número de vezes que o filtro é aplicado        |
| `sigma`       | Desvio padrão do kernel Gaussiano              |
| `repetições`  | Número de repetições para calcular a média     |

**Exemplo — parâmetros usados nos testes do projeto:**

```bash
# Serial
./serial image_4096x4096.png 100 1.0 5

# OpenMP com 4 threads
OMP_NUM_THREADS=4 ./openmp image_4096x4096.png 100 1.0 5

# Tiling com 8 threads
OMP_NUM_THREADS=8 ./tiling image_4096x4096.png 100 1.0 5
```

A saída é salva automaticamente em `output_serial.png`, `output_openmp_Nthreads.png` ou `output_tiling_Nthreads.png`.

---

## Relatório e gráficos

O relatório detalha a implementação, análise de desempenho e conclusões.
O script `gerar_graficos.py` gera os gráficos de tempo, speedup e eficiência a partir dos arquivos de saída dos benchmarks.
Por escolha, o codigo LaTeX do relatório não está incluído no repositório, pois necessitaria de dependências adicionais para compilar (como `pdflatex` e pacotes LaTeX). O relatório finalizado está disponível em PDF.

---

## Testes.txt

O arquivo `testes.txt` contém o comando exato usado para cada teste, incluindo os parâmetros e o número de threads. Ele serve como referência para os dados de desempenho apresentados no relatório.

## Resultados.txt

O arquivo `resultados.txt` contém os tempos de execução médios para cada teste, organizados por implementação (serial, OpenMP, tiling) e número de threads. Ele é a base para os gráficos de desempenho e análise apresentados no relatório.

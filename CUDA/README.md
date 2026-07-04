# Etapa 3 — CUDA / GPGPU

Implementação do desfoque Gaussiano iterativo em GPU com CUDA, para comparar com as versões serial, OpenMP e MPI.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `gaussian.cuh` | protótipos das funções de host |
| `gaussian.cu`  | funções de host: I/O (stb), geração do kernel |
| `cuda.cu`      | os dois kernels de device + benchmark (main) |
| `Makefile`     | compila com `nvcc` |
| `testes.txt`   | bateria de comandos usada nos testes |

## Duas versões implementadas

1. **`global`** — uma thread por pixel, lendo os vizinhos direto da memória global. Versão direta, sem otimização de memória.
2. **`shared`** — *tiling*: cada bloco carrega um tile de 16×16 pixels mais a moldura de halo na **memória compartilhada** (`__shared__`), sincroniza com `__syncthreads()` e convolui lendo da SRAM rápida. O kernel Gaussiano fica em **memória constante** (`__constant__`). É o análogo GPU da troca de halos do MPI.

## Pré-requisitos (no notebook com a GTX 1650)

- Driver NVIDIA atualizado
- CUDA Toolkit (`nvcc --version` deve funcionar)
- `nvidia-smi` deve listar a GTX 1650

## Compilar

```bash
cd CUDA/
make
```

> O `Makefile` usa `-arch=sm_75` (Turing, arquitetura da GTX 1650). Se a compilação reclamar da arquitetura, rode `nvcc --help | grep gpu-architecture` para ver as suportadas pela sua versão do toolkit.

## Executar

```
./cuda <imagem.png> [iteracoes] [kernel_size] [sigma] [repeticoes] [modo]
```

Todos os argumentos além da imagem são opcionais. Defaults (iguais à bateria das etapas anteriores): `iteracoes=1000`, `kernel_size=3`, `sigma=1.0`, `repeticoes=5`, `modo=shared`.

```bash
# versão shared, todos os defaults
./cuda ../image_4096x4096.png

# versão global, explícito
./cuda ../image_4096x4096.png 1000 3 1.0 5 global
```

A saída é salva em `output_cuda_shared.png` ou `output_cuda_global.png`.

O tempo é medido com **eventos CUDA** (`cudaEventRecord`), cronometrando apenas o laço de iterações na GPU — as cópias host↔device iniciais ficam de fora, para comparar de forma justa com o tempo de cálculo das outras etapas.

## Corretude

Para validar, compare a saída CUDA com a do serial (mesmos parâmetros):

```bash
# na raiz do repo, com output_serial.png e output_cuda_shared.png gerados
python3 -c "
import numpy as np; from PIL import Image
a=np.array(Image.open('OpenMP/output_serial.png').convert('L'),dtype=float)
b=np.array(Image.open('CUDA/output_cuda_shared.png').convert('L'),dtype=float)
d=np.abs(a-b)
print('diff max:', d.max(), '| EQM:', (d**2).mean(), '| pixels diferentes:', int((d>0).sum()))
"
```

> Observação: diferente do MPI, aqui pode aparecer diferença **mínima** (1 nível de cinza em pouquíssimos pixels). Isso é esperado: a GPU pode usar *fused multiply-add* (FMA) e ordem de soma diferente da CPU, gerando arredondamento levemente distinto no último bit do float. Uma diferença máxima de 0 ou 1, com EQM próximo de zero, indica implementação correta.

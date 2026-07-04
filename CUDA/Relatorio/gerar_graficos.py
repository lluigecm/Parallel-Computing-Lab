"""
Gera os graficos para o relatorio do benchmark de Gaussian Blur - CUDA (Etapa 3)
DEC107 - Processamento Paralelo
Dados: notebook Intel Core i5-11300H + NVIDIA GeForce GTX 1650 (Turing, sm_75)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

os.makedirs('figuras', exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
})

# ── Dados (media de 5 repeticoes, 1000 iteracoes, kernel 3x3, sigma 1.0) ───────

sizes = ['512x512', '1024x1024', '2048x2048', '4096x4096']

serial_times = [0.627442, 2.817162, 11.447778, 44.709906]

# CUDA: tempo do laco de kernel (cudaEvent), sem copias H<->D
cuda_global = [0.034512, 0.110099, 0.438412, 1.760318]
cuda_shared = [0.038919, 0.143264, 0.580867, 2.364881]

# Melhores tempos de CPU neste hardware (8 threads / 8 processos)
omp_8t = [0.286326, 1.050046, 3.449537, 13.016647]
mpi_8p = [0.551877, 2.195850, 8.079189, 32.741364]

# Baterias completas de CPU para o grafico do 4096
omp_4096 = [26.734225, 21.369637, 13.016647]   # 2T, 4T, 8T
mpi_4096 = [74.617300, 52.280315, 32.741364]   # 2P, 4P, 8P

sp_global = [serial_times[i] / cuda_global[i] for i in range(4)]
sp_shared = [serial_times[i] / cuda_shared[i] for i in range(4)]

# ── Paleta (mesma dos relatorios anteriores) ──────────────────────────────────

CUDA_G_COLOR = '#2ca02c'   # verde  (global)
CUDA_S_COLOR = '#9467bd'   # roxo   (shared)
OMP_COLOR    = '#1f77b4'   # azul
MPI_COLOR    = '#d62728'   # vermelho
SERIAL_COLOR = '#7f7f7f'   # cinza


def annotate_bars(ax, bars, fmt='{:.1f}x', offset=3, fs=8.5):
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset), textcoords='offset points',
            ha='center', va='bottom', fontsize=fs,
        )


# ── Grafico 1: Speedup CUDA (global vs shared) vs serial ──────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(sizes))
w = 0.38
b1 = ax.bar(x - w/2, sp_global, w, label='CUDA global', color=CUDA_G_COLOR, alpha=0.9)
b2 = ax.bar(x + w/2, sp_shared, w, label='CUDA shared', color=CUDA_S_COLOR, alpha=0.9)
annotate_bars(ax, b1)
annotate_bars(ax, b2)
ax.set_xlabel('Tamanho da imagem')
ax.set_ylabel('Speedup vs serial')
ax.set_title('Speedup do CUDA sobre o serial')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_ylim(0, max(sp_global) + 4)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('figuras/speedup_cuda.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 2: Tempo de execucao de todas as abordagens (escala log) ──────────

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(sizes))
series = [
    ('Serial',      serial_times, SERIAL_COLOR, 'o'),
    ('OpenMP 8T',   omp_8t,       OMP_COLOR,    's'),
    ('MPI 8P',      mpi_8p,       MPI_COLOR,    '^'),
    ('CUDA global', cuda_global,  CUDA_G_COLOR, 'D'),
    ('CUDA shared', cuda_shared,  CUDA_S_COLOR, 'v'),
]
for nome, vals, cor, mk in series:
    ax.plot(x, vals, marker=mk, color=cor, linewidth=1.8, markersize=6, label=nome)
ax.set_yscale('log')
ax.set_xlabel('Tamanho da imagem')
ax.set_ylabel('Tempo medio (s) - escala log')
ax.set_title('Tempo de execucao - todas as abordagens (melhor config. de CPU)')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, which='both')
plt.tight_layout()
plt.savefig('figuras/tempo_todas.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 3: Tempo de execucao - 4096x4096 (barras, log) ────────────────────

labels = ['Serial',
          'OMP\n2T', 'OMP\n4T', 'OMP\n8T',
          'MPI\n2P', 'MPI\n4P', 'MPI\n8P',
          'CUDA\nglobal', 'CUDA\nshared']
vals = [serial_times[3], *omp_4096, *mpi_4096, cuda_global[3], cuda_shared[3]]
cores = [SERIAL_COLOR] + [OMP_COLOR]*3 + [MPI_COLOR]*3 + [CUDA_G_COLOR, CUDA_S_COLOR]

fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.bar(range(len(vals)), vals, color=cores, alpha=0.9)
ax.set_yscale('log')
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel('Configuracao')
ax.set_ylabel('Tempo medio (s) - escala log')
ax.set_title('Tempo de execucao - 4096x4096')
ax.grid(True, linestyle='--', alpha=0.4, axis='y', which='both')
legend_elements = [Patch(facecolor=SERIAL_COLOR, label='Serial'),
                   Patch(facecolor=OMP_COLOR,    label='OpenMP'),
                   Patch(facecolor=MPI_COLOR,    label='MPI'),
                   Patch(facecolor=CUDA_G_COLOR, label='CUDA global'),
                   Patch(facecolor=CUDA_S_COLOR, label='CUDA shared')]
ax.legend(handles=legend_elements, fontsize=8.5)
for bar in bars:
    ax.annotate(f'{bar.get_height():.2f}s',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('figuras/tempo_4096.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 4: Global vs Shared variando o tamanho do kernel (2048x2048) ──────

kernels = ['3x3', '5x5', '7x7', '9x9', '11x11']
kx = np.arange(len(kernels))
g_kernel = [0.452, 0.852, 1.249, 2.075, 2.800]
s_kernel = [0.585, 0.917, 1.218, 2.096, 2.758]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(kx, g_kernel, marker='D', color=CUDA_G_COLOR, linewidth=1.8,
        markersize=6, label='CUDA global')
ax.plot(kx, s_kernel, marker='v', color=CUDA_S_COLOR, linewidth=1.8,
        markersize=6, label='CUDA shared')
ax.set_xlabel('Tamanho do kernel')
ax.set_ylabel('Tempo medio (s)')
ax.set_title('Global vs Shared variando o kernel - 2048x2048')
ax.set_xticks(kx)
ax.set_xticklabels(kernels)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/kernel_global_shared.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 5: Escalabilidade (log-log) - CUDA vs CPU ─────────────────────────

pixel_counts = [512**2, 1024**2, 2048**2, 4096**2]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(pixel_counts, serial_times, marker='o', color=SERIAL_COLOR,
          linewidth=1.8, markersize=6, label='Serial')
ax.loglog(pixel_counts, omp_8t, marker='s', color=OMP_COLOR,
          linewidth=1.8, markersize=6, label='OpenMP 8T')
ax.loglog(pixel_counts, cuda_global, marker='D', color=CUDA_G_COLOR,
          linewidth=1.8, markersize=6, label='CUDA global')
ax.loglog(pixel_counts, cuda_shared, marker='v', color=CUDA_S_COLOR,
          linewidth=1.8, markersize=6, label='CUDA shared')
ax.set_xlabel('Numero de pixels (N^2)')
ax.set_ylabel('Tempo medio (s)')
ax.set_title('Escalabilidade - Tempo vs tamanho da imagem')
ax.set_xticks(pixel_counts)
ax.get_xaxis().set_major_formatter(
    mticker.FuncFormatter(lambda val, _: f'{int(val**0.5)}^2')
)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, which='both')
plt.tight_layout()
plt.savefig('figuras/escalabilidade.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 6: Eficiencia de banda de memoria da GPU ──────────────────────────
# banda alcancada = trafego minimo por iteracao (ler + escrever a imagem) / tempo
# eficiencia = banda alcancada / pico (GDDR5 da GTX 1650 = 128 GB/s)
# Estimativa: e um piso (assume que o cache L2 absorve o reuso dos vizinhos).

N_pix     = [512, 1024, 2048, 4096]
PICO_BW   = 128e9        # 128 GB/s, GDDR5 (8 Gbps x 128 bits)

def eficiencia_banda(tempos_1000it):
    ef = []
    for i, t in enumerate(tempos_1000it):
        t_iter  = t / 1000.0
        trafego = 2 * N_pix[i]**2 * 4     # le 1x + escreve 1x, 4 bytes/pixel
        bw      = trafego / t_iter
        ef.append(bw / PICO_BW * 100)
    return ef

ef_global = eficiencia_banda(cuda_global)
ef_shared = eficiencia_banda(cuda_shared)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
xg = np.arange(len(sizes))
ax.axhline(100, color='#aaaaaa', linestyle='--', linewidth=1.2, label='Pico (128 GB/s)')
ax.plot(xg, ef_global, marker='D', color=CUDA_G_COLOR, linewidth=1.8,
        markersize=6, label='CUDA global')
ax.plot(xg, ef_shared, marker='v', color=CUDA_S_COLOR, linewidth=1.8,
        markersize=6, label='CUDA shared')
for x, y in zip(xg, ef_global):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points', xytext=(0, 7),
                ha='center', fontsize=8.5, color=CUDA_G_COLOR)
for x, y in zip(xg, ef_shared):
    ax.annotate(f'{y:.0f}%', (x, y), textcoords='offset points', xytext=(0, -14),
                ha='center', fontsize=8.5, color=CUDA_S_COLOR)
ax.set_xlabel('Tamanho da imagem')
ax.set_ylabel('Eficiência de banda (% do pico)')
ax.set_title('Eficiência de banda de memória da GPU (estimativa, piso)')
ax.set_xticks(xg)
ax.set_xticklabels(sizes)
ax.set_ylim(0, 110)
ax.legend(loc='lower left')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/eficiencia_banda_gpu.pdf', bbox_inches='tight')
plt.close()

print('Graficos gerados em figuras/')

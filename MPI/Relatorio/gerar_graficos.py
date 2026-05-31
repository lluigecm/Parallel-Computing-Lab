"""
Gera os graficos para o relatorio do benchmark de Gaussian Blur - MPI
DEC107 - Processamento Paralelo
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

# ── Dados ─────────────────────────────────────────────────────────────────────

sizes   = ['512x512', '1024x1024', '2048x2048', '4096x4096']
procs   = [2, 4, 8, 16]

serial_times = [0.853337, 3.457297, 13.740706, 61.483692]

# indices: [tamanho][proc_idx]
openmp_times = [
    [0.621649, 0.376205, 0.389969, 0.292832],   # 512
    [2.653334, 1.390496, 1.526054, 1.036950],   # 1024
    [10.056233, 5.378655, 6.042983, 4.107041],  # 2048
    [41.044222, 24.023297, 24.791501, 15.798529], # 4096
]
mpi_times = [
    [0.449172, 0.235773, 0.293238, 0.214107],   # 512
    [1.828533, 1.134228, 1.123115, 0.767939],   # 1024
    [7.296503, 3.884077, 4.395790, 3.078722],   # 2048
    [29.088768, 16.255987, 17.344438, 11.181296], # 4096
]

# Speedup e eficiencia
omp_sp  = [[serial_times[i] / t for t in openmp_times[i]] for i in range(4)]
mpi_sp  = [[serial_times[i] / t for t in mpi_times[i]]    for i in range(4)]
omp_eff = [[omp_sp[i][j] / procs[j] * 100 for j in range(4)] for i in range(4)]
mpi_eff = [[mpi_sp[i][j] / procs[j] * 100 for j in range(4)] for i in range(4)]

# ── Paleta ────────────────────────────────────────────────────────────────────

COLORS       = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
MARKERS      = ['o', 's', '^', 'D']
MPI_COLOR    = '#d62728'
OMP_COLOR    = '#1f77b4'
SERIAL_COLOR = '#7f7f7f'


def annotate_bars(ax, bars, fmt='{:.2f}x', offset=3):
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset), textcoords='offset points',
            ha='center', va='bottom', fontsize=8.5,
        )


# ── Grafico 1: Speedup MPI ────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(procs, procs, '--', color='#aaaaaa', linewidth=1.2, label='Ideal', zorder=1)
for i in range(4):
    ax.plot(procs, mpi_sp[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de processos')
ax.set_ylabel('Speedup')
ax.set_title('Speedup - MPI')
ax.set_xticks(procs)
ax.set_ylim(0, max(procs) + 1)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/speedup_mpi.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 2: Eficiencia MPI ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axhline(100, color='#aaaaaa', linestyle='--', linewidth=1.2, label='Ideal')
for i in range(4):
    ax.plot(procs, mpi_eff[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de processos')
ax.set_ylabel('Eficiencia (%)')
ax.set_title('Eficiencia - MPI')
ax.set_xticks(procs)
ax.set_ylim(0, 120)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/eficiencia_mpi.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 3: Comparacao MPI vs OpenMP — 16 processos ───────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(sizes))
w = 0.35
omp_sp16 = [omp_sp[i][3] for i in range(4)]
mpi_sp16 = [mpi_sp[i][3] for i in range(4)]
b1 = ax.bar(x - w/2, omp_sp16, w, label='OpenMP', color=OMP_COLOR, alpha=0.85)
b2 = ax.bar(x + w/2, mpi_sp16, w, label='MPI',    color=MPI_COLOR, alpha=0.85)
annotate_bars(ax, b1)
annotate_bars(ax, b2)
ax.set_xlabel('Tamanho da imagem')
ax.set_ylabel('Speedup')
ax.set_title('Speedup com 16 processos/threads - MPI vs OpenMP')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('figuras/comparacao_16.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 4: Tempo de execucao — 4096x4096 ─────────────────────────────────

labels_4096 = ['Serial',
               'OMP\n2T', 'OMP\n4T', 'OMP\n8T', 'OMP\n16T',
               'MPI\n2P', 'MPI\n4P', 'MPI\n8P', 'MPI\n16P']
vals_4096 = [serial_times[3], *openmp_times[3], *mpi_times[3]]
bar_colors = [SERIAL_COLOR] + [OMP_COLOR] * 4 + [MPI_COLOR] * 4

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.bar(range(9), vals_4096, color=bar_colors, alpha=0.85)
ax.set_xticks(range(9))
ax.set_xticklabels(labels_4096, fontsize=9)
ax.set_xlabel('Configuracao')
ax.set_ylabel('Tempo medio (s)')
ax.set_title('Tempo de execucao - 4096x4096')
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
legend_elements = [Patch(facecolor=SERIAL_COLOR, label='Serial'),
                   Patch(facecolor=OMP_COLOR,    label='OpenMP'),
                   Patch(facecolor=MPI_COLOR,    label='MPI')]
ax.legend(handles=legend_elements)
for bar in bars:
    ax.annotate(
        f'{bar.get_height():.1f}s',
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3), textcoords='offset points',
        ha='center', va='bottom', fontsize=8.5,
    )
plt.tight_layout()
plt.savefig('figuras/tempo_4096.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 5: Escalabilidade (log-log) ──────────────────────────────────────

pixel_counts = [512**2, 1024**2, 2048**2, 4096**2]
omp_16t = [openmp_times[i][3] for i in range(4)]
mpi_16p = [mpi_times[i][3]    for i in range(4)]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(pixel_counts, serial_times, marker='o', color=SERIAL_COLOR,
          linewidth=1.8, markersize=6, label='Serial')
ax.loglog(pixel_counts, omp_16t, marker='s', color=OMP_COLOR,
          linewidth=1.8, markersize=6, label='OpenMP 16T')
ax.loglog(pixel_counts, mpi_16p, marker='^', color=MPI_COLOR,
          linewidth=1.8, markersize=6, label='MPI 16P')
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

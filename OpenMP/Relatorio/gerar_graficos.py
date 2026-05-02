"""
Gera os graficos para o relatorio do benchmark de Gaussian Blur
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
threads = [2, 4, 8, 16]

serial_times = [0.081402, 0.330098, 1.330545, 5.561620]

# indices: [tamanho][thread_idx]
openmp_times = [
    [0.058977, 0.031080, 0.039596, 0.030358],   # 512
    [0.259517, 0.139225, 0.147411, 0.102504],   # 1024
    [0.964820, 0.488918, 0.590665, 0.468031],   # 2048
    [4.331640, 2.336690, 2.380236, 1.609739],   # 4096
]
tiling_times = [
    [0.064505, 0.033555, 0.030666, 0.034276],   # 512
    [0.284514, 0.133881, 0.099971, 0.095700],   # 1024
    [1.046301, 0.555603, 0.428715, 0.377533],   # 2048
    [4.523856, 2.559652, 1.774299, 1.472678],   # 4096
]

# Speedup e eficiencia
omp_sp  = [[serial_times[i] / t for t in openmp_times[i]] for i in range(4)]
til_sp  = [[serial_times[i] / t for t in tiling_times[i]] for i in range(4)]
omp_eff = [[omp_sp[i][j] / threads[j] * 100 for j in range(4)] for i in range(4)]
til_eff = [[til_sp[i][j] / threads[j] * 100 for j in range(4)] for i in range(4)]

# ── Paleta ────────────────────────────────────────────────────────────────────

COLORS       = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
MARKERS      = ['o', 's', '^', 'D']
OMP_COLOR    = '#1f77b4'
TILING_COLOR = '#ff7f0e'
SERIAL_COLOR = '#7f7f7f'


def annotate_bars(ax, bars, fmt='{:.2f}x', offset=3):
    for bar in bars:
        ax.annotate(
            fmt.format(bar.get_height()),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset), textcoords='offset points',
            ha='center', va='bottom', fontsize=8.5,
        )


# ── Grafico 1: Speedup OpenMP ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(threads, threads, '--', color='#aaaaaa', linewidth=1.2, label='Ideal', zorder=1)
for i in range(4):
    ax.plot(threads, omp_sp[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de threads')
ax.set_ylabel('Speedup')
ax.set_title('Speedup - OpenMP')
ax.set_xticks(threads)
ax.set_ylim(0, max(threads) + 1)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/speedup_openmp.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 2: Speedup Tiling ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(threads, threads, '--', color='#aaaaaa', linewidth=1.2, label='Ideal', zorder=1)
for i in range(4):
    ax.plot(threads, til_sp[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de threads')
ax.set_ylabel('Speedup')
ax.set_title('Speedup - Tiling')
ax.set_xticks(threads)
ax.set_ylim(0, max(threads) + 1)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/speedup_tiling.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 3: Eficiencia OpenMP ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axhline(100, color='#aaaaaa', linestyle='--', linewidth=1.2, label='Ideal')
for i in range(4):
    ax.plot(threads, omp_eff[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de threads')
ax.set_ylabel('Eficiencia (%)')
ax.set_title('Eficiencia - OpenMP')
ax.set_xticks(threads)
ax.set_ylim(0, 115)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/eficiencia_openmp.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 4: Eficiencia Tiling ──────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.axhline(100, color='#aaaaaa', linestyle='--', linewidth=1.2, label='Ideal')
for i in range(4):
    ax.plot(threads, til_eff[i], marker=MARKERS[i], color=COLORS[i],
            linewidth=1.8, markersize=6, label=sizes[i])
ax.set_xlabel('Numero de threads')
ax.set_ylabel('Eficiencia (%)')
ax.set_title('Eficiencia - Tiling')
ax.set_xticks(threads)
ax.set_ylim(0, 115)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('figuras/eficiencia_tiling.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 5: Comparacao OpenMP vs Tiling — 16 threads ──────────────────────

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(sizes))
w = 0.35
omp_sp16 = [omp_sp[i][3] for i in range(4)]
til_sp16 = [til_sp[i][3] for i in range(4)]
b1 = ax.bar(x - w/2, omp_sp16, w, label='OpenMP', color=OMP_COLOR,    alpha=0.85)
b2 = ax.bar(x + w/2, til_sp16, w, label='Tiling',  color=TILING_COLOR, alpha=0.85)
annotate_bars(ax, b1)
annotate_bars(ax, b2)
ax.set_xlabel('Tamanho da imagem')
ax.set_ylabel('Speedup')
ax.set_title('Speedup com 16 threads - OpenMP vs Tiling')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('figuras/comparacao_16t.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 6: Tempo de execucao — 4096x4096 ─────────────────────────────────

labels_4096 = ['Serial',
               'OMP\n2T', 'OMP\n4T', 'OMP\n8T', 'OMP\n16T',
               'Tiling\n2T', 'Tiling\n4T', 'Tiling\n8T', 'Tiling\n16T']
vals_4096 = [serial_times[3], *openmp_times[3], *tiling_times[3]]
bar_colors = [SERIAL_COLOR] + [OMP_COLOR] * 4 + [TILING_COLOR] * 4

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
                   Patch(facecolor=TILING_COLOR,  label='Tiling')]
ax.legend(handles=legend_elements)
for bar in bars:
    ax.annotate(
        f'{bar.get_height():.2f}s',
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 3), textcoords='offset points',
        ha='center', va='bottom', fontsize=8.5,
    )
plt.tight_layout()
plt.savefig('figuras/tempo_4096.pdf', bbox_inches='tight')
plt.close()

# ── Grafico 7: Escalabilidade (log-log) ──────────────────────────────────────

pixel_counts = [512**2, 1024**2, 2048**2, 4096**2]
omp_16t = [openmp_times[i][3] for i in range(4)]
til_16t = [tiling_times[i][3] for i in range(4)]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(pixel_counts, serial_times, marker='o', color=SERIAL_COLOR,
          linewidth=1.8, markersize=6, label='Serial')
ax.loglog(pixel_counts, omp_16t, marker='s', color=OMP_COLOR,
          linewidth=1.8, markersize=6, label='OpenMP 16T')
ax.loglog(pixel_counts, til_16t, marker='^', color=TILING_COLOR,
          linewidth=1.8, markersize=6, label='Tiling 16T')
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
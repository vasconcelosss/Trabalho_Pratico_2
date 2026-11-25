import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from christofides import christofides
from features import _projetar
from heuristics import nearest_neighbor, convex_hull_insertion
from instancias import INSTANCIAS, _achar_arquivo
from matching import emparelhamento_perfeito_minimo
from mst import prim, vertices_grau_impar
from twice_around import twice_around_the_tree
from utils import ler_instancia_tsplib_completa


def _desenhar_rota(ax, P, rota, cor, estilo='-', largura=1.6, alpha=0.9, label=None):
    xs = [P[v, 0] for v in rota] + [P[rota[0], 0]]
    ys = [P[v, 1] for v in rota] + [P[rota[0], 1]]
    ax.plot(xs, ys, linestyle=estilo, color=cor, linewidth=largura, alpha=alpha, label=label)


def _desenhar_pontos(ax, P, n, destaque=None):
    ax.scatter(P[:, 0], P[:, 1], s=28, color='#2c3e50', zorder=4)
    if destaque:
        idx = list(destaque)
        ax.scatter(P[idx, 0], P[idx, 1], s=55, color='#e74c3c', zorder=5,
                   label='Grau ímpar (MST)')
    if n <= 35:
        for i in range(n):
            ax.annotate(str(i), (P[i, 0], P[i, 1]), fontsize=6.5,
                        xytext=(3, 3), textcoords='offset points', zorder=6)


def plotar_instancia(nome, pasta='data', pasta_fig='results/figuras/grafos'):
    entrada = next((e for e in INSTANCIAS if e['nome'] == nome), None)
    base = entrada.get('base', nome) if entrada else nome
    d = ler_instancia_tsplib_completa(_achar_arquivo(pasta, base))

    if entrada and 'k' in entrada:
        k = entrada['k']
        coords = {i: d['coords'][i] for i in range(k)}
        from utils import calcular_matriz_distancias
        dist = calcular_matriz_distancias(coords, k, d['tipo'])
        n = k
    else:
        coords, dist, n = d['coords'], d['distancias'], d['n']

    if not coords or len(coords) < n:
        print(f"  {nome}: sem coordenadas 2D para desenho estático.")
        return

    otimo = entrada['otimo'] if entrada else None
    P = _projetar(coords, n, d['tipo'])

    arestas_mst = prim(dist, n)
    impares = set(vertices_grau_impar(arestas_mst, n))
    arestas_match, _ = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    rota_tw, c_tw, _ = twice_around_the_tree(dist, n)
    rota_ch, c_ch, _, _ = christofides(dist, n)
    rota_nn, c_nn = nearest_neighbor(dist, n)
    rota_hu, c_hu = convex_hull_insertion(dist, n, coords)

    def _gap(c):
        return f" (+{(c - otimo) / otimo * 100:.1f}%)" if otimo else ""

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # MST + Matching (estrutura do Christofides)
    ax = axes[0, 0]
    for u, v, _ in arestas_mst:
        ax.plot([P[u, 0], P[v, 0]], [P[u, 1], P[v, 1]],
                color='#2980b9', linewidth=1.5, alpha=0.85,
                label='MST (Prim)' if (u, v) == arestas_mst[0][:2] else None)
    for idx, (u, v) in enumerate(arestas_match):
        ax.plot([P[u, 0], P[v, 0]], [P[u, 1], P[v, 1]],
                color='#e67e22', linestyle='--', linewidth=1.6, alpha=0.95,
                label='Matching Mínimo' if idx == 0 else None)
    _desenhar_pontos(ax, P, n, destaque=impares)
    ax.set_title('1. MST + Vértices Ímpares + Matching (Christofides)', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='best')
    ax.grid(alpha=0.2)

    #  Twice-around-the-tree
    ax = axes[0, 1]
    _desenhar_rota(ax, P, rota_tw, '#8e44ad', label=f'Twice-around = {c_tw}{_gap(c_tw)}')
    _desenhar_pontos(ax, P, n)
    ax.set_title(f'2. Twice-around-the-tree: custo {c_tw}{_gap(c_tw)}', fontsize=9.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.2)

    # Christofides
    ax = axes[1, 0]
    _desenhar_rota(ax, P, rota_ch, '#16a085', label=f'Christofides = {c_ch}{_gap(c_ch)}')
    _desenhar_pontos(ax, P, n)
    ax.set_title(f'3. Christofides: custo {c_ch}{_gap(c_ch)}', fontsize=9.5)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.2)

    # Nearest Neighbor vs Convex Hull
    ax = axes[1, 1]
    _desenhar_rota(ax, P, rota_nn, '#c0392b', estilo='--', largura=1.3, alpha=0.8,
                   label=f'Nearest Neighbor = {c_nn}{_gap(c_nn)}')
    _desenhar_rota(ax, P, rota_hu, '#27ae60', estilo='-', largura=1.8, alpha=0.9,
                   label=f'Convex Hull Ins. = {c_hu}{_gap(c_hu)}')
    _desenhar_pontos(ax, P, n)
    ax.set_title('4. Nearest Neighbor vs Convex Hull Insertion', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='best')
    ax.grid(alpha=0.2)

    otimo_txt = f" | Ótimo = {otimo}" if otimo else ""
    fig.suptitle(f"Visualização Comparativa: {nome} (n={n}, {d['tipo']}{otimo_txt})", fontsize=12)
    fig.tight_layout()
    os.makedirs(pasta_fig, exist_ok=True)
    saida = os.path.join(pasta_fig, f"{nome}_comparativo.png")
    fig.savefig(saida, dpi=160)
    plt.close(fig)
    print(f"  Salvo: {saida}")


if __name__ == '__main__':
    alvos = sys.argv[1:] or ['ulysses16', 'ulysses22', 'bays29', 'berlin52']
    for nome in alvos:
        plotar_instancia(nome)
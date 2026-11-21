"""
Testes para o algoritmo Twice-around-the-tree.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from twice_around import twice_around_the_tree
from mst import prim, peso_total_mst
from utils import calcular_custo_rota


def _montar_matriz(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xd = coords[i][0] - coords[j][0]
            yd = coords[i][1] - coords[j][1]
            dist[i][j] = math.sqrt(xd * xd + yd * yd)
    return dist, n


def test_todos_vertices_na_rota():
    """A rota deve conter todos os N vértices."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo, _ = twice_around_the_tree(dist, n)

    assert len(rota) == n, f"Esperado {n} vértices na rota, obteve {len(rota)}"
    print(f"  PASSOU: Rota tem {len(rota)} vértices (esperado {n})")


def test_sem_vertices_repetidos():
    """Cada vértice deve aparecer exatamente uma vez na rota."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, _, _ = twice_around_the_tree(dist, n)

    assert len(rota) == len(set(rota)), \
        f"Vértices repetidos! Rota: {rota}"
    print(f"  PASSOU: Sem vértices repetidos na rota")


def test_custo_no_maximo_2_vezes_mst():
    """
    Propriedade teórica: custo da rota ≤ 2 * peso(MST).
    Isso confirma que a implementação respeita a garantia de aproximação.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (2, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo_rota, arestas_mst = twice_around_the_tree(dist, n)
    peso_mst = peso_total_mst(arestas_mst)

    assert custo_rota <= 2 * peso_mst + 1e-9, \
        f"Custo {custo_rota:.4f} > 2 * MST {2*peso_mst:.4f} — violação da garantia!"
    print(f"  PASSOU: custo({custo_rota:.4f}) ≤ 2 * MST({peso_mst:.4f}) = {2*peso_mst:.4f}")


def test_custo_no_maximo_2_vezes_mst_linha():
    """Mesma verificação para uma linha reta de 5 pontos."""
    coords = [(i, 0) for i in range(5)]
    dist, n = _montar_matriz(coords)

    rota, custo_rota, arestas_mst = twice_around_the_tree(dist, n)
    peso_mst = peso_total_mst(arestas_mst)

    assert custo_rota <= 2 * peso_mst + 1e-9, \
        f"Custo {custo_rota:.4f} > 2 * MST {2*peso_mst:.4f}"
    print(f"  PASSOU: linha reta — custo({custo_rota:.4f}) ≤ 2*MST({2*peso_mst:.4f})")


def test_conjunto_vertices_completo():
    """Verifica que o conjunto de vértices visitados é exatamente {0, 1, ..., N-1}."""
    coords = [(0, 0), (1, 2), (3, 1), (2, 4), (4, 3)]
    dist, n = _montar_matriz(coords)

    rota, _, _ = twice_around_the_tree(dist, n)

    assert set(rota) == set(range(n)), \
        f"Vértices faltando ou errados! Esperado {set(range(n))}, obteve {set(rota)}"
    print(f"  PASSOU: Conjunto de vértices visitados é {set(range(n))}")


def test_custo_calculado_corretamente():
    """
    Verifica que o custo retornado pela função é consistente
    com o cálculo manual usando calcular_custo_rota.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo_retornado, _ = twice_around_the_tree(dist, n)
    custo_manual = calcular_custo_rota(rota, dist)

    assert abs(custo_retornado - custo_manual) < 1e-9, \
        f"Custo retornado ({custo_retornado:.6f}) != custo calculado ({custo_manual:.6f})"
    print(f"  PASSOU: Custo retornado ({custo_retornado:.4f}) é consistente")


def test_custo_no_maximo_2_vezes_otimo_quadrado():
    """
    Para o quadrado unitário, a solução ótima é percorrer os 4 lados = custo 4.
    O Twice-around-the-tree deve produzir custo ≤ 2 * 4 = 8.
    (Na prática tende a ser muito melhor.)
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo, _ = twice_around_the_tree(dist, n)
    custo_otimo = 4.0  # quadrado unitário: 4 lados de comprimento 1

    assert custo <= 2 * custo_otimo + 1e-9, \
        f"Custo {custo:.4f} > 2 * ótimo {2*custo_otimo:.4f}"
    print(f"  PASSOU: custo({custo:.4f}) ≤ 2 * ótimo({custo_otimo:.1f}) = {2*custo_otimo:.1f}")


if __name__ == "__main__":
    print("=" * 55)
    print("Testes: Twice-around-the-tree")
    print("=" * 55)

    testes = [
        test_todos_vertices_na_rota,
        test_sem_vertices_repetidos,
        test_custo_no_maximo_2_vezes_mst,
        test_custo_no_maximo_2_vezes_mst_linha,
        test_conjunto_vertices_completo,
        test_custo_calculado_corretamente,
        test_custo_no_maximo_2_vezes_otimo_quadrado,
    ]

    passou = 0
    falhou = 0
    for teste in testes:
        print(f"\nExecutando: {teste.__name__}")
        try:
            teste()
            passou += 1
        except AssertionError as e:
            print(f"  FALHOU: {e}")
            falhou += 1
        except Exception as e:
            print(f"  ERRO INESPERADO: {type(e).__name__}: {e}")
            falhou += 1

    print(f"\n{'='*55}")
    print(f"Resultado: {passou}/{len(testes)} passaram, {falhou} falharam")
    print("=" * 55)

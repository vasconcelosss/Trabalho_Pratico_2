"""
Testes para o algoritmo de Christofides.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from christofides import christofides
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


def test_todos_vertices_visitados():
    """A rota deve conter todos os N vértices exatamente uma vez."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo, _, _ = christofides(dist, n)

    assert len(rota) == n, f"Esperado {n} vértices, obteve {len(rota)}"
    assert len(set(rota)) == n, "Vértice repetido na rota!"
    assert set(rota) == set(range(n)), f"Vértice(s) faltando: {set(range(n)) - set(rota)}"
    print(f"  PASSOU: Rota hamiltoniana com {n} vértices distintos")


def test_custo_consistente():
    """Custo retornado deve bater com cálculo manual."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    rota, custo_ret, _, _ = christofides(dist, n)
    custo_manual = calcular_custo_rota(rota, dist)

    assert abs(custo_ret - custo_manual) < 1e-9, \
        f"Custo retornado ({custo_ret:.6f}) != custo calculado ({custo_manual:.6f})"
    print(f"  PASSOU: Custo = {custo_ret:.4f} (consistente)")


def test_garantia_3_sobre_2():
    """
    Propriedade teórica: custo(Christofides) ≤ 3/2 * peso(MST) + ε.

    Mais precisamente: custo ≤ peso(MST) + peso(matching)
    e peso(matching) ≤ 1/2 * OPT.

    Verificamos que custo ≤ 3/2 * peso(MST), que é uma consequência de
    peso(MST) ≤ OPT e peso(matching) ≤ 1/2 * OPT ≤ 1/2 * peso(MST)... *

    * Na verdade isso só vale quando MST = OPT, mas serve como sanidade.

    O teste mais correto é: custo ≤ peso(MST) + peso(matching), que sempre vale.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (2, 2)]
    dist, n = _montar_matriz(coords)

    rota, custo, arestas_mst, peso_match = christofides(dist, n)
    peso_mst = peso_total_mst(arestas_mst)

    # O custo do Christofides deve ser ≤ peso(MST) + peso(matching)
    limite_teorico = peso_mst + peso_match
    assert custo <= limite_teorico + 1e-9, \
        f"Custo ({custo:.4f}) > MST+matching ({limite_teorico:.4f})!"
    print(f"  PASSOU: custo({custo:.4f}) ≤ MST({peso_mst:.4f}) + matching({peso_match:.4f}) = {limite_teorico:.4f}")


def test_christofides_melhor_ou_igual_twice_around():
    """
    O Christofides não garante ser SEMPRE melhor que o Twice-around-the-tree,
    mas sua razão de aproximação teórica é 3/2 contra 2.
    Este teste verifica que o custo é razoável (≤ 2 * peso_MST).
    """
    from twice_around import twice_around_the_tree

    coords = [(0, 0), (1, 2), (3, 1), (2, 4), (4, 3), (1, 5)]
    dist, n = _montar_matriz(coords)

    rota_c, custo_c, _, _ = christofides(dist, n)
    rota_t, custo_t, arestas_mst = twice_around_the_tree(dist, n)
    peso_mst = peso_total_mst(arestas_mst)

    # Ambos devem estar dentro da garantia teórica de 2 * MST
    assert custo_c <= 2 * peso_mst + 1e-9, \
        f"Christofides ({custo_c:.4f}) > 2*MST ({2*peso_mst:.4f})"
    assert custo_t <= 2 * peso_mst + 1e-9, \
        f"Twice-around ({custo_t:.4f}) > 2*MST ({2*peso_mst:.4f})"
    print(f"  PASSOU: Christofides={custo_c:.4f}, Twice-around={custo_t:.4f}, 2*MST={2*peso_mst:.4f}")


def test_instancia_maior():
    """Verifica que o algoritmo funciona corretamente para n=10."""
    import random
    random.seed(42)
    coords = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(10)]
    dist, n = _montar_matriz(coords)

    rota, custo, _, _ = christofides(dist, n)

    assert len(rota) == n
    assert len(set(rota)) == n
    assert set(rota) == set(range(n))
    assert custo > 0
    print(f"  PASSOU: n=10 → rota válida, custo={custo:.2f}")


def test_circuito_euleriano_interno():
    """Testa diretamente a função de circuito euleriano com um caso simples."""
    from christofides import _circuito_euleriano

    # Triângulo: 0-1-2-0 (3 arestas, todas com grau 2 → euleriano)
    adj = {
        0: [1, 2],
        1: [0, 2],
        2: [1, 0],
    }
    circuito = _circuito_euleriano(adj, 3)

    # Verifica que é um circuito válido (começa e termina no mesmo vértice)
    assert circuito[0] == circuito[-1], "Circuito euleriano deve ser fechado!"
    assert len(circuito) == 4, f"Triângulo deve ter 3 arestas → 4 vértices no circuito"
    print(f"  PASSOU: Circuito euleriano no triângulo = {circuito}")


if __name__ == "__main__":
    print("=" * 55)
    print("Testes: Algoritmo de Christofides")
    print("=" * 55)

    testes = [
        test_todos_vertices_visitados,
        test_custo_consistente,
        test_garantia_3_sobre_2,
        test_christofides_melhor_ou_igual_twice_around,
        test_instancia_maior,
        test_circuito_euleriano_interno,
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
            print(f"  ERRO: {type(e).__name__}: {e}")
            falhou += 1

    print(f"\n{'='*55}")
    print(f"Resultado: {passou}/{len(testes)} passaram, {falhou} falharam")
    print("=" * 55)

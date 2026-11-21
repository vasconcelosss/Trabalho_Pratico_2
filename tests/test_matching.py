"""
Testes para o emparelhamento perfeito de peso mínimo.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mst import prim, vertices_grau_impar
from matching import emparelhamento_perfeito_minimo


def _montar_matriz(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xd = coords[i][0] - coords[j][0]
            yd = coords[i][1] - coords[j][1]
            dist[i][j] = math.sqrt(xd * xd + yd * yd)
    return dist, n


def test_todos_impares_emparelhados():
    """
    Todos os vértices de grau ímpar devem aparecer exatamente uma vez
    no emparelhamento (definição de emparelhamento perfeito).
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 2.0)]
    dist, n = _montar_matriz(coords)
    arestas_mst = prim(dist, n)

    impares = vertices_grau_impar(arestas_mst, n)
    arestas_m, _ = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    # Conta quantas vezes cada vértice ímpar aparece no emparelhamento
    contagem = {v: 0 for v in impares}
    for u, v in arestas_m:
        contagem[u] += 1
        contagem[v] += 1

    for v, c in contagem.items():
        assert c == 1, f"Vértice {v} aparece {c} vezes no emparelhamento (esperado 1)"

    print(f"  PASSOU: Todos os {len(impares)} vértices ímpares emparelhados exatamente 1 vez")


def test_numero_de_pares():
    """
    O número de pares no emparelhamento deve ser exatamente
    metade do número de vértices de grau ímpar.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 2.0)]
    dist, n = _montar_matriz(coords)
    arestas_mst = prim(dist, n)

    impares = vertices_grau_impar(arestas_mst, n)
    arestas_m, _ = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    esperado = len(impares) // 2
    assert len(arestas_m) == esperado, \
        f"Esperado {esperado} pares, obteve {len(arestas_m)}"
    print(f"  PASSOU: {len(arestas_m)} pares no emparelhamento (esperado {esperado})")


def test_peso_consistente():
    """
    O peso retornado deve ser igual à soma das distâncias dos pares.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)
    arestas_mst = prim(dist, n)

    arestas_m, peso_retornado = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    peso_manual = sum(dist[u][v] for u, v in arestas_m)
    assert abs(peso_retornado - peso_manual) < 1e-9, \
        f"Peso retornado ({peso_retornado:.6f}) != calculado ({peso_manual:.6f})"
    print(f"  PASSOU: Peso do emparelhamento = {peso_retornado:.4f} (consistente)")


def test_emparelhamento_minimo_caso_simples():
    """
    4 vértices em linha: 0-1-2-3 com distância 1 entre consecutivos.
    A MST é a linha, e os vértices de grau ímpar são 0 e 3.
    O emparelhamento ótimo é {(0,3)} com peso 3,
    mas dependendo da MST pode variar.
    O importante é que seja um emparelhamento válido.
    """
    coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
    dist, n = _montar_matriz(coords)
    arestas_mst = prim(dist, n)

    impares = vertices_grau_impar(arestas_mst, n)
    arestas_m, peso = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    # Verifica que é um emparelhamento válido (cada vértice ímpar exatamente uma vez)
    vertices_emparelhados = []
    for u, v in arestas_m:
        vertices_emparelhados.extend([u, v])

    assert sorted(vertices_emparelhados) == sorted(impares), \
        f"Emparelhamento inválido! Esperado {sorted(impares)}, obteve {sorted(vertices_emparelhados)}"
    print(f"  PASSOU: Emparelhamento válido para linha reta, peso = {peso:.4f}")


def test_sem_vertices_duplicados_no_emparelhamento():
    """
    Nenhum vértice deve aparecer em dois pares diferentes do emparelhamento.
    Isso é a definição de 'emparelhamento' (matching).
    """
    coords = [(0, 0), (1, 2), (3, 1), (2, 4), (4, 3), (1, 5)]
    dist, n = _montar_matriz(coords)
    arestas_mst = prim(dist, n)

    arestas_m, _ = emparelhamento_perfeito_minimo(dist, n, arestas_mst)

    todos_vertices = []
    for u, v in arestas_m:
        todos_vertices.extend([u, v])

    assert len(todos_vertices) == len(set(todos_vertices)), \
        f"Vértice duplicado no emparelhamento! {todos_vertices}"
    print(f"  PASSOU: Nenhum vértice aparece em dois pares diferentes")


if __name__ == "__main__":
    print("=" * 55)
    print("Testes: Emparelhamento Perfeito de Peso Mínimo")
    print("=" * 55)

    testes = [
        test_todos_impares_emparelhados,
        test_numero_de_pares,
        test_peso_consistente,
        test_emparelhamento_minimo_caso_simples,
        test_sem_vertices_duplicados_no_emparelhamento,
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

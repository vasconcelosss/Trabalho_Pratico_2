"""
Testes para o algoritmo de Prim (MST).

Usamos instâncias simples criadas manualmente para verificar a corretude
antes de usar as instâncias reais da TSPLIB.
"""
import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mst import prim, peso_total_mst, graus_vertices, vertices_grau_impar


# --- Funções auxiliares para montar matrizes de teste ---

def _distancia(c1, c2):
    xd = c1[0] - c2[0]
    yd = c1[1] - c2[1]
    return math.sqrt(xd * xd + yd * yd)

def _montar_matriz(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dist[i][j] = _distancia(coords[i], coords[j])
    return dist, n


# --- Testes ---

def test_numero_de_arestas():
    """A MST de N vértices deve ter exatamente N-1 arestas."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    arestas = prim(dist, n)

    assert len(arestas) == n - 1, f"Esperado {n-1} arestas, obteve {len(arestas)}"
    print(f"  PASSOU: MST tem {len(arestas)} arestas (esperado {n-1})")


def test_todos_vertices_conectados():
    """Todos os vértices devem aparecer na MST."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)

    arestas = prim(dist, n)

    vertices_presentes = set()
    for u, v, _ in arestas:
        vertices_presentes.add(u)
        vertices_presentes.add(v)

    todos = set(range(n))
    faltando = todos - vertices_presentes
    assert not faltando, f"Vértices faltando na MST: {faltando}"
    print(f"  PASSOU: Todos os {n} vértices aparecem na MST")


def test_sem_ciclos():
    """
    Verificação via BFS: se com N-1 arestas todos os N vértices são
    alcançáveis a partir do vértice 0, então não há ciclos.
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)
    arestas = prim(dist, n)

    # Monta lista de adjacência
    adj = {v: [] for v in range(n)}
    for u, v, _ in arestas:
        adj[u].append(v)
        adj[v].append(u)

    # BFS a partir do vértice 0
    visitados = {0}
    fila = [0]
    while fila:
        atual = fila.pop(0)
        for viz in adj[atual]:
            if viz not in visitados:
                visitados.add(viz)
                fila.append(viz)

    assert visitados == set(range(n)), "BFS não alcançou todos os vértices!"
    print(f"  PASSOU: MST é conexa e sem ciclos ({n-1} arestas, {n} vértices)")


def test_peso_quadrado_unitario():
    """
    Quadrado unitário: a MST deve usar 3 lados de comprimento 1.0
    (não as diagonais de comprimento sqrt(2)).
    Peso total esperado: 3.0
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dist, n = _montar_matriz(coords)
    arestas = prim(dist, n)
    peso = peso_total_mst(arestas)

    assert abs(peso - 3.0) < 1e-9, f"Peso esperado 3.0, obtido {peso:.6f}"
    print(f"  PASSOU: Peso da MST no quadrado unitário = {peso:.4f} (esperado 3.0)")


def test_graus_somam_dois_vezes_numero_arestas():
    """
    Teorema do Aperto de Mão: a soma dos graus deve ser 2*(N-1).
    """
    coords = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]
    dist, n = _montar_matriz(coords)
    arestas = prim(dist, n)

    graus = graus_vertices(arestas, n)
    soma_graus = sum(graus)
    esperado = 2 * (n - 1)

    assert soma_graus == esperado, \
        f"Soma dos graus esperada {esperado}, obtida {soma_graus}"
    print(f"  PASSOU: Soma dos graus = {soma_graus} = 2*(n-1) = {esperado}")


def test_vertices_grau_impar_numero_par():
    """
    O número de vértices de grau ímpar sempre deve ser par
    (consequência do Teorema do Aperto de Mão).
    """
    for coords in [
        [(0, 0), (1, 0), (1, 1), (0, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],  # linha reta
        [(0, 0), (1, 1), (2, 0), (1, -1)],
    ]:
        dist, n = _montar_matriz(coords)
        arestas = prim(dist, n)
        impares = vertices_grau_impar(arestas, n)

        assert len(impares) % 2 == 0, \
            f"Número de vértices de grau ímpar deve ser par! Obtido: {len(impares)}"

    print(f"  PASSOU: Número de vértices de grau ímpar sempre é par")


def test_estrela():
    """
    Instância estrela: vértice central conectado a 4 periféricos.
    Todos os periféricos equidistantes do centro (distância 5).
    A MST deve ser a estrela com peso total 20.
    """
    # vértice 0 = centro em (5,5), periféricos nas quatro direções
    coords = [(5, 5), (0, 5), (10, 5), (5, 0), (5, 10)]
    dist, n = _montar_matriz(coords)
    arestas = prim(dist, n)

    peso = peso_total_mst(arestas)
    assert abs(peso - 20.0) < 1e-9, f"Peso esperado 20.0, obtido {peso:.6f}"

    # Verifica que todas as arestas partem do vértice 0 (ou chegam nele)
    for u, v, _ in arestas:
        assert u == 0 or v == 0, \
            f"Aresta ({u},{v}) não envolve o centro — MST errada!"
    print(f"  PASSOU: Instância estrela, peso = {peso:.1f} (esperado 20.0)")


def test_linha_reta():
    """
    5 vértices em linha reta: 0--1--2--3--4 com espaçamento 1.
    A MST deve ser exatamente essa linha, com peso 4.
    """
    coords = [(i, 0) for i in range(5)]
    dist, n = _montar_matriz(coords)
    arestas = prim(dist, n)

    peso = peso_total_mst(arestas)
    assert abs(peso - 4.0) < 1e-9, f"Peso esperado 4.0, obtido {peso:.6f}"
    print(f"  PASSOU: Linha reta de 5 pontos, peso = {peso:.1f} (esperado 4.0)")


# --- Ponto de entrada ---

if __name__ == "__main__":
    print("=" * 50)
    print("Testes: Algoritmo de Prim (MST)")
    print("=" * 50)

    testes = [
        test_numero_de_arestas,
        test_todos_vertices_conectados,
        test_sem_ciclos,
        test_peso_quadrado_unitario,
        test_graus_somam_dois_vezes_numero_arestas,
        test_vertices_grau_impar_numero_par,
        test_estrela,
        test_linha_reta,
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
            print(f"  ERRO INESPERADO: {e}")
            falhou += 1

    print(f"\n{'='*50}")
    print(f"Resultado: {passou}/{len(testes)} passaram, {falhou} falharam")
    print("=" * 50)

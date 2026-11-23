import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mst import prim, adjacencia_mst
from utils import calcular_custo_rota


def twice_around_the_tree(distancias, n):
    """
    Calcula uma rota aproximada para o TSP com garantia de fator 2.

    Implementação via DFS em preordem na MST, que equivale a percorrer
    o circuito euleriano na MST duplicada e aplicar shortcut.

    Parâmetros:
        distancias: matriz n×n de distâncias
        n         : número de vértices

    Retorna:
        rota       : lista de vértices na ordem de visita (sem repetições)
        custo      : custo total da rota (volta ao vértice inicial incluída)
        arestas_mst: arestas da MST utilizada
    """
    # Passo 1: Construir a MST com Prim
    arestas_mst = prim(distancias, n)

    # Passo 2: Converter para lista de adjacência
    adj = adjacencia_mst(arestas_mst, n)

    # Passo 3: DFS em preordem na MST
    # O percurso em preordem corresponde a dar a volta na árvore e pular vértices
    # já visitados — exatamente o shortcut sobre o circuito euleriano na MST duplicada.
    rota = []
    visitados = set()

    def dfs(v):
        visitados.add(v)
        rota.append(v)
        for vizinho, _ in adj[v]:
            if vizinho not in visitados:
                dfs(vizinho)

    # Iniciamos do vértice 0 (poderia ser qualquer vértice)
    # Para n ≤ 75 a recursão não causa problemas de pilha
    dfs(0)

    # Passo 4: Calcular o custo (calcular_custo_rota já inclui a volta para rota[0])
    custo = calcular_custo_rota(rota, distancias)

    return rota, custo, arestas_mst

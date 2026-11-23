import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mst import prim, adjacencia_mst, vertices_grau_impar
from matching import emparelhamento_perfeito_minimo
from utils import calcular_custo_rota


def _circuito_euleriano(adj_multi, n):
    """
    Encontra um circuito euleriano no multigrafo usando o algoritmo de Hierholzer.

    O multigrafo é representado como um dicionário de listas de vizinhos.
    Todo vértice deve ter grau par para que o circuito exista.

    Complexidade: O(m) onde m é o número de arestas.
    """
    # Verificação: todos os vértices com grau > 0 devem ter grau par
    for v in range(n):
        if len(adj_multi[v]) % 2 != 0:
            raise ValueError(f"Vértice {v} tem grau ímpar: {len(adj_multi[v])}")

    # Hierholzer: começa do vértice 0, usa pilha para backtracking
    # Escolhe um vértice de partida que tenha arestas
    inicio = 0
    for v in range(n):
        if adj_multi[v]:
            inicio = v
            break

    pilha = [inicio]
    circuito = []

    while pilha:
        v = pilha[-1]
        if adj_multi[v]:
            # Remove a aresta (v, vizinho) do multigrafo
            vizinho = adj_multi[v].pop()
            # Remove a aresta simétrica (vizinho, v)
            adj_multi[vizinho].remove(v)
            pilha.append(vizinho)
        else:
            # Backtrack: v não tem mais arestas, faz parte do circuito
            circuito.append(pilha.pop())

    return circuito


def christofides(distancias, n):
    """
    Algoritmo de Christofides: aproximação 3/2 para o TSP métrico.

    Parâmetros:
        distancias: matriz n×n de distâncias
        n         : número de vértices

    Retorna:
        rota       : lista de vértices (sem repetições, ciclo hamiltoniano)
        custo      : custo total da rota
        arestas_mst: arestas da MST usada
        peso_match : peso do emparelhamento usado
    """
    # Passo 1: Construir a MST
    arestas_mst = prim(distancias, n)

    # Passo 2: Identificar vértices de grau ímpar na MST
    impares = vertices_grau_impar(arestas_mst, n)

    # Passo 3: Emparelhamento perfeito de peso mínimo nos vértices ímpares
    arestas_match, peso_match = emparelhamento_perfeito_minimo(
        distancias, n, arestas_mst
    )

    # Passo 4: Combinar MST + emparelhamento → multigrafo H
    # Usamos listas (não conjuntos) para permitir arestas paralelas (multigrafo)
    adj_multi = {v: [] for v in range(n)}

    for u, v, _ in arestas_mst:
        adj_multi[u].append(v)
        adj_multi[v].append(u)

    for u, v in arestas_match:
        adj_multi[u].append(v)
        adj_multi[v].append(u)

    # Passo 5: Circuito euleriano em H (todo vértice tem grau par agora)
    # Fazemos uma cópia porque _circuito_euleriano modifica o dicionário
    adj_copia = {v: list(vizinhos) for v, vizinhos in adj_multi.items()}
    circuito = _circuito_euleriano(adj_copia, n)

    # Passo 6: Converter circuito euleriano em rota hamiltoniana (shortcut)
    # Percorre o circuito e mantém apenas a primeira ocorrência de cada vértice
    visitados = set()
    rota = []
    for v in circuito:
        if v not in visitados:
            visitados.add(v)
            rota.append(v)

    # Passo 7: Calcular custo (inclui retorno ao vértice inicial)
    custo = calcular_custo_rota(rota, distancias)

    return rota, custo, arestas_mst, peso_match

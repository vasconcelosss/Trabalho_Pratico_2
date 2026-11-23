import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import networkx as nx
    NETWORKX_DISPONIVEL = True
except ImportError:
    NETWORKX_DISPONIVEL = False

from mst import vertices_grau_impar


def emparelhamento_perfeito_minimo(distancias, n, arestas_mst):
    """
    Encontra o emparelhamento perfeito de peso mínimo entre os vértices
    de grau ímpar da MST.

    Usa NetworkX (algoritmo de Blossom) para garantir o resultado ótimo.
    Uma abordagem gulosa simples não seria suficiente — poderia dar o par
    "mais próximo" localmente sem garantir o mínimo global.

    Parâmetros:
        distancias: matriz n×n de distâncias
        n         : número total de vértices
        arestas_mst: arestas da MST (para identificar os vértices de grau ímpar)

    Retorna:
        arestas_matching: lista de pares (u, v) do emparelhamento
        peso_matching   : soma dos pesos das arestas do emparelhamento
    """
    if not NETWORKX_DISPONIVEL:
        raise ImportError(
            "NetworkX não encontrado. Instale com: pip install networkx"
        )

    # Passo 1: Identifica os vértices de grau ímpar na MST
    impares = vertices_grau_impar(arestas_mst, n)

    # Pelo Teorema do Aperto de Mão, sempre há um número par de vértices ímpares
    assert len(impares) % 2 == 0, "Número de vértices ímpares deve ser par!"

    if len(impares) == 0:
        # Caso raro: nenhum vértice ímpar (a MST já é euleriana — só possível para n=1)
        return [], 0

    # Passo 2: Monta o grafo completo com apenas os vértices de grau ímpar
    # Usamos pesos negativos porque NetworkX implementa emparelhamento de PESO MÁXIMO.
    # Para obter peso mínimo, negamos todos os pesos.
    G = nx.Graph()
    G.add_nodes_from(impares)

    for i in range(len(impares)):
        for j in range(i + 1, len(impares)):
            u = impares[i]
            v = impares[j]
            peso = distancias[u][v]
            # Negar o peso para transformar em problema de peso máximo
            G.add_edge(u, v, weight=-peso)

    # Passo 3: Calcula o emparelhamento perfeito de peso máximo (= mínimo com pesos negativos)
    # max_weight_matching com maxcardinality=True garante que todos os vértices são emparelhados
    matching = nx.max_weight_matching(G, maxcardinality=True)

    # Passo 4: Converte para lista de pares e calcula o peso real (positivo)
    arestas_matching = list(matching)
    peso_matching = sum(distancias[u][v] for u, v in arestas_matching)

    return arestas_matching, peso_matching

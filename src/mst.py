"""
Algoritmo de Prim para Árvore Geradora Mínima (MST).

Referência teórica principal: Cormen et al., "Algoritmos: Teoria e Prática",
Capítulo 23 - Árvores Geradoras Mínimas.

O algoritmo de Prim é um algoritmo guloso que constrói a MST incrementalmente.
A ideia central é manter um conjunto S de vértices já incluídos na árvore e,
a cada passo, adicionar o vértice fora de S que tem a aresta mais leve
conectando-o a algum vértice em S.

Implementação com complexidade O(n²), adequada para instâncias pequenas.
Para instâncias grandes, seria necessário usar um heap de prioridades (O(m log n)),
mas para o nosso conjunto de teste (n ≤ 75) a versão simples é suficiente.
"""


def prim(distancias, n):
    """
    Encontra a Árvore Geradora Mínima pelo algoritmo de Prim.

    Parâmetros:
        distancias: lista de listas [n][n] com as distâncias entre vértices
        n         : número de vértices

    Retorna:
        lista de tuplas (u, v, peso) representando as n-1 arestas da MST
    """
    # chave[v]: menor distância de um vértice já na árvore até v
    # Inicialmente infinito para todos, exceto o vértice de partida (0)
    chave = [float('inf')] * n
    chave[0] = 0

    # predecessor[v]: de qual vértice da árvore v foi alcançado
    predecessor = [-1] * n

    # na_arvore[v]: indica se v já foi adicionado à MST
    na_arvore = [False] * n

    for _ in range(n):
        # Escolhe o vértice de menor chave que ainda não está na árvore
        # Essa busca linear é O(n); com heap seria O(log n)
        u = -1
        for v in range(n):
            if not na_arvore[v]:
                if u == -1 or chave[v] < chave[u]:
                    u = v

        # Marca u como parte da árvore
        na_arvore[u] = True

        # Atualiza as chaves dos vizinhos de u que ainda não estão na árvore
        for v in range(n):
            if not na_arvore[v] and distancias[u][v] < chave[v]:
                chave[v] = distancias[u][v]
                predecessor[v] = u

    # Monta a lista de arestas (o vértice 0 não tem predecessor)
    arestas = []
    for v in range(1, n):
        u = predecessor[v]
        peso = distancias[u][v]
        arestas.append((u, v, peso))

    return arestas


def peso_total_mst(arestas_mst):
    """Soma os pesos de todas as arestas da MST."""
    return sum(peso for _, _, peso in arestas_mst)


def graus_vertices(arestas_mst, n):
    """
    Calcula o grau de cada vértice na MST.

    O grau de um vértice é o número de arestas incidentes nele.
    Isso é necessário no algoritmo de Christofides para identificar
    os vértices de grau ímpar.
    """
    graus = [0] * n
    for u, v, _ in arestas_mst:
        graus[u] += 1
        graus[v] += 1
    return graus


def vertices_grau_impar(arestas_mst, n):
    """
    Retorna a lista de vértices com grau ímpar na MST.

    Pelo Teorema do Aperto de Mão (Handshaking Lemma), a soma de todos os
    graus é sempre par (igual a 2 * número de arestas). Por isso, o número
    de vértices com grau ímpar é sempre par — o que garante que o
    emparelhamento perfeito entre eles sempre existe.

    Referência: Kleinberg & Tardos, "Algorithm Design", Cap. 2.
    """
    graus = graus_vertices(arestas_mst, n)
    return [v for v in range(n) if graus[v] % 2 == 1]


def adjacencia_mst(arestas_mst, n):
    """
    Monta a lista de adjacência da MST.
    Usada nos algoritmos de percurso (DFS/BFS) para Twice-around-the-tree.
    """
    adj = {v: [] for v in range(n)}
    for u, v, peso in arestas_mst:
        adj[u].append((v, peso))
        adj[v].append((u, peso))
    return adj

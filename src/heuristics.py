import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import calcular_custo_rota


def cold_start(distancias, n):
    """
    rota ingenua 0 -> 1 -> 2 -> ... -> n-1 -> 0.
    """
    if n == 0:
        return [], 0
    rota = list(range(n))
    custo = calcular_custo_rota(rota, distancias)
    return rota, custo


def nearest_neighbor(distancias, n, inicio=0):
    """
    algoritmo guloso
    """
    if n == 0:
        return [], 0

    visitados = [False] * n
    visitados[inicio] = True
    rota = [inicio]
    atual = inicio

    for _ in range(n - 1):
        melhor_v = -1
        melhor_dist = float('inf')
        linha = distancias[atual]
        for v in range(n):
            if not visitados[v] and linha[v] < melhor_dist:
                melhor_dist = linha[v]
                melhor_v = v
        visitados[melhor_v] = True
        rota.append(melhor_v)
        atual = melhor_v

    custo = calcular_custo_rota(rota, distancias)
    return rota, custo


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


# Adaptação de: https://www.geeksforgeeks.org/dsa/convex-hull-monotone-chain-algorithm/
def findConvexHull_indices(coords, n):
    if n < 3:
        return list(range(n))

    sortedPoints = sorted((coords[i][0], coords[i][1], i) for i in range(n))

    hull = []
    for p in sortedPoints:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    lowerSize = len(hull) + 1
    for i in range(n - 2, -1, -1):
        p = sortedPoints[i]
        while len(hull) >= lowerSize and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)

    hull.pop()
    return [p[2] for p in hull]


def convex_hull_insertion(distancias, n, coords):
    if n == 0:
        return [], 0
    if n <= 3:
        rota = list(range(n))
        return rota, calcular_custo_rota(rota, distancias)
    if not coords or len(coords) < n:
        raise ValueError("Coordenadas 2D completas são necessárias para o Fecho Convexo.")

    sub_rota = findConvexHull_indices(coords, n)
    na_rota = set(sub_rota)
    internos = [v for v in range(n) if v not in na_rota]

    while internos:
        melhor_v = -1
        melhor_pos = -1
        melhor_delta = float('inf')
        m = len(sub_rota)

        for v in internos:
            for i in range(m):
                u = sub_rota[i]
                w = sub_rota[(i + 1) % m]
                delta = distancias[u][v] + distancias[v][w] - distancias[u][w]
                if delta < melhor_delta:
                    melhor_delta = delta
                    melhor_v = v
                    melhor_pos = i + 1

        sub_rota.insert(melhor_pos, melhor_v)
        internos.remove(melhor_v)

    custo = calcular_custo_rota(sub_rota, distancias)
    return sub_rota, custo
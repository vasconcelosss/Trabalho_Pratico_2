import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from heuristics import cold_start, nearest_neighbor, convex_hull_insertion, fecho_convexo
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


def _validar_rota(rota, n):
    assert len(rota) == n
    assert set(rota) == set(range(n))


def test_cold_start():
    coords = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    dist, n = _montar_matriz(coords)
    rota, custo = cold_start(dist, n)
    _validar_rota(rota, n)
    assert rota == [0, 1, 2, 3]
    assert abs(custo - 4.0) < 1e-9


def test_nearest_neighbor():
    coords = {0: (0, 0), 1: (10, 0), 2: (1, 0), 3: (2, 0)}
    dist, n = _montar_matriz(coords)
    rota, custo = nearest_neighbor(dist, n, inicio=0)
    _validar_rota(rota, n)
    assert rota == [0, 2, 3, 1]
    assert abs(custo - calcular_custo_rota(rota, dist)) < 1e-9


def test_fecho_convexo_com_ponto_interno():
    coords = {0: (0, 0), 1: (4, 0), 2: (4, 4), 3: (0, 4), 4: (2, 2)}
    hull = fecho_convexo(coords, 5)
    assert set(hull) == {0, 1, 2, 3}
    assert 4 not in hull


def test_convex_hull_insertion_quadrado_com_centro():
    coords = {0: (0, 0), 1: (4, 0), 2: (4, 4), 3: (0, 4), 4: (2, 2)}
    dist, n = _montar_matriz(coords)
    rota, custo = convex_hull_insertion(dist, n, coords)
    _validar_rota(rota, n)
    assert abs(custo - calcular_custo_rota(rota, dist)) < 1e-9


def test_convex_hull_insertion_colinear():
    coords = {i: (float(i), 0.0) for i in range(6)}
    dist, n = _montar_matriz(coords)
    rota, custo = convex_hull_insertion(dist, n, coords)
    _validar_rota(rota, n)
    assert abs(custo - 10.0) < 1e-9


if __name__ == "__main__":
    for t in [
        test_cold_start,
        test_nearest_neighbor,
        test_fecho_convexo_com_ponto_interno,
        test_convex_hull_insertion_quadrado_com_centro,
        test_convex_hull_insertion_colinear,
    ]:
        t()
        print(f"PASSOU: {t.__name__}")
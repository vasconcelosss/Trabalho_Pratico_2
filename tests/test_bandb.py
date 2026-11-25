import sys
import os
import math
import random
import itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from branch_and_bound import branch_and_bound, executar_warm_starts
from utils import calcular_custo_rota


def _montar_matriz(coords):
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            xd = coords[i][0] - coords[j][0]
            yd = coords[i][1] - coords[j][1]
            dist[i][j] = int(math.sqrt(xd * xd + yd * yd) + 0.5)
    return dist, n


def _forca_bruta(dist, n):
    melhor = float('inf')
    for perm in itertools.permutations(range(1, n)):
        rota = [0] + list(perm)
        c = calcular_custo_rota(rota, dist)
        if c < melhor:
            melhor = c
    return melhor


def test_otimalidade_contra_forca_bruta():
    rng = random.Random(123)
    for n in [4, 6, 8]:
        coords = {i: (rng.uniform(0, 100), rng.uniform(0, 100)) for i in range(n)}
        dist, _ = _montar_matriz(coords)
        opt_fb = _forca_bruta(dist, n)

        res_dfs = branch_and_bound(dist, n, estrategia='dfs')
        res_bfs = branch_and_bound(dist, n, estrategia='best_first')

        assert res_dfs['status'] == 'OK'
        assert res_bfs['status'] == 'OK'
        assert res_dfs['custo'] == opt_fb
        assert res_bfs['custo'] == opt_fb
        assert len(set(res_dfs['rota'])) == n


def test_cinco_estrategias_warm_start_mesmo_otimo():
    rng = random.Random(42)
    n = 9
    coords = {i: (rng.uniform(0, 500), rng.uniform(0, 500)) for i in range(n)}
    dist, _ = _montar_matriz(coords)
    opt_fb = _forca_bruta(dist, n)

    resultados = executar_warm_starts(dist, n, coords=coords, timeout=30.0)
    assert len(resultados) == 5

    for nome, dados in resultados.items():
        assert dados['status'] == 'OK', f"Falhou em {nome}: {dados}"
        assert dados['custo_otimo'] == opt_fb, f"{nome}: {dados['custo_otimo']} != {opt_fb}"
        assert dados['nos_expandidos'] >= 1
        assert dados['nos_podados'] >= 0


def test_timeout_preserva_estatisticas():
    rng = random.Random(7)
    n = 35
    coords = {i: (rng.uniform(0, 1000), rng.uniform(0, 1000)) for i in range(n)}
    dist, _ = _montar_matriz(coords)

    res = branch_and_bound(dist, n, timeout=0.05, estrategia='dfs')
    assert res['status'] == 'TIMEOUT'
    assert res['custo'] is None
    assert res['tempo_s'] is None
    assert res['nos_expandidos'] > 0


if __name__ == "__main__":
    for t in [
        test_otimalidade_contra_forca_bruta,
        test_cinco_estrategias_warm_start_mesmo_otimo,
        test_timeout_preserva_estatisticas,
    ]:
        t()
        print(f"PASSOU: {t.__name__}")
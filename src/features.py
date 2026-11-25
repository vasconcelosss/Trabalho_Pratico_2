import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from heuristics import findConvexHull_indices
from instancias import INSTANCIAS, _achar_arquivo
from mst import prim, graus_vertices
from utils import (ler_instancia_tsplib_completa, calcular_matriz_distancias,
                   _geo_para_radianos)

NAN = float('nan')

COLUNAS = [
    'instancia', 'n', 'faixa', 'tipo', 'coords_reais',
    'razao_aspecto', 'frac_hull', 'entropia_espacial', 'clark_evans',
    'cv_nn', 'grau_max_mst', 'frac_folhas_mst', 'frac_impares_mst',
    'cv_arestas_mst', 'razao_mst_max',
]


def _cv(valores):
    v = np.asarray(valores, dtype=float)
    m = v.mean()
    return float(v.std() / m) if m > 0 else NAN


def _projetar(coords, n, tipo):
    """devolve array em coordenadas planas."""
    if tipo == 'GEO':
        lat = np.array([_geo_para_radianos(coords[i][0]) for i in range(n)])
        lon = np.array([_geo_para_radianos(coords[i][1]) for i in range(n)])
        return np.column_stack([lon * math.cos(lat.mean()), lat])
    return np.array([coords[i] for i in range(n)], dtype=float)


def _features_geometricas(coords, n, tipo):
    vazio = {k: NAN for k in ('razao_aspecto', 'frac_hull', 'entropia_espacial', 'clark_evans')}
    if not coords or len(coords) < n or n < 3:
        return vazio

    P = _projetar(coords, n, tipo)
    largura = float(P[:, 0].max() - P[:, 0].min())
    altura = float(P[:, 1].max() - P[:, 1].min())
    menor, maior = min(largura, altura), max(largura, altura)
    razao = maior / menor if menor > 0 else NAN

    hull = findConvexHull_indices({i: (P[i, 0], P[i, 1]) for i in range(n)}, n)
    frac_hull = len(hull) / n

    g = max(2, int(math.ceil(math.sqrt(n))))
    H, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=g)
    p = H.ravel() / n
    p = p[p > 0]
    entropia = float(-(p * np.log(p)).sum() / math.log(g * g))

    dif = P[:, None, :] - P[None, :, :]
    D = np.sqrt((dif ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    nn_medio = float(D.min(axis=1).mean())
    area = largura * altura
    clark = nn_medio / (0.5 * math.sqrt(area / n)) if area > 0 else NAN

    return {'razao_aspecto': razao, 'frac_hull': frac_hull,
            'entropia_espacial': entropia, 'clark_evans': clark}


def _features_matriz(distancias, n):
    D = np.array(distancias, dtype=float)
    np.fill_diagonal(D, np.inf)
    cv_nn = _cv(D.min(axis=1))

    arestas = prim(distancias, n)
    graus = np.array(graus_vertices(arestas, n))
    pesos = np.array([w for _, _, w in arestas], dtype=float)
    media = float(pesos.mean()) if len(pesos) else 0.0
    return {
        'cv_nn': cv_nn,
        'grau_max_mst': int(graus.max()),
        'frac_folhas_mst': float((graus == 1).sum() / n),
        'frac_impares_mst': float((graus % 2 == 1).sum() / n),
        'cv_arestas_mst': _cv(pesos) if len(pesos) else NAN,
        'razao_mst_max': float(pesos.max() / media) if media > 0 else NAN,
    }


def extrair_features(coords, distancias, n, tipo='EUC_2D'):
    """Dicionário com todas as features de uma instância."""
    f = _features_geometricas(coords, n, tipo)
    f.update(_features_matriz(distancias, n))
    return f


# ---------------------------------------------------------------------------
# Execução sobre o conjunto de instâncias definido em instancias.py
# ---------------------------------------------------------------------------
def _carregar_com_meta(pasta):
    cache, saida = {}, []
    for e in INSTANCIAS:
        base = e.get('base', e['nome'])
        if base not in cache:
            try:
                cache[base] = ler_instancia_tsplib_completa(_achar_arquivo(pasta, base))
            except FileNotFoundError as err:
                print(f"AVISO: {err}; instância ignorada")
                continue
        d = cache[base]
        if 'k' in e:
            k = e['k']
            coords = {i: d['coords'][i] for i in range(k)}
            dist = calcular_matriz_distancias(coords, k, d['tipo'])
            n = k
        else:
            coords, dist, n = d['coords'], d['distancias'], d['n']
        saida.append({'nome': e['nome'], 'n': n, 'faixa': e['faixa'], 'tipo': d['tipo'],
                      'coords_reais': d['coords_reais'], 'coords': coords, 'distancias': dist})
    return saida


def gerar_csv(pasta='data', arquivo_saida='results/features.csv'):
    os.makedirs(os.path.dirname(arquivo_saida) or '.', exist_ok=True)
    with open(arquivo_saida, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for inst in _carregar_com_meta(pasta):
            feats = extrair_features(inst['coords'], inst['distancias'], inst['n'], inst['tipo'])
            linha = {'instancia': inst['nome'], 'n': inst['n'], 'faixa': inst['faixa'],
                     'tipo': inst['tipo'], 'coords_reais': inst['coords_reais']}
            for k, v in feats.items():
                linha[k] = round(v, 6) if isinstance(v, float) and not math.isnan(v) else (
                    'NA' if isinstance(v, float) else v)
            w.writerow(linha)
            print(f"  {inst['nome']:<14} n={inst['n']:<3} hull={linha['frac_hull']} "
                  f"grau_max_mst={linha['grau_max_mst']} cv_nn={linha['cv_nn']}")
    print(f"\nFeatures salvas em: {arquivo_saida}")


if __name__ == '__main__':
    gerar_csv(sys.argv[1] if len(sys.argv) > 1 else 'data',
              sys.argv[2] if len(sys.argv) > 2 else 'results/features.csv')
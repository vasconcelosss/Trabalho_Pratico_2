import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from utils import (ler_instancia_tsplib_completa, calcular_matriz_distancias,
                   calcular_custo_rota)
from heuristics import nearest_neighbor

INSTANCIAS = [
    # --- faixa principal (N <= 40) ---
    {'nome': 'burma14',   'otimo': 3323, 'faixa': 'principal'},   # GEO
    {'nome': 'ulysses16', 'otimo': 6859, 'faixa': 'principal'},   # GEO
    {'nome': 'gr17',      'otimo': 2085, 'faixa': 'principal'},   # EXPLICIT
    {'nome': 'gr21',      'otimo': 2707, 'faixa': 'principal'},   # EXPLICIT
    {'nome': 'ulysses22', 'otimo': 7013, 'faixa': 'principal'},   # GEO
    {'nome': 'gr24',      'otimo': 1272, 'faixa': 'principal'},   # EXPLICIT
    {'nome': 'fri26',     'otimo': 937,  'faixa': 'principal'},   # EXPLICIT
    {'nome': 'bayg29',    'otimo': 1610, 'faixa': 'principal'},   # GEO
    {'nome': 'bays29',    'otimo': 2020, 'faixa': 'principal'},   # GEO
    {'nome': 'berlin52_k32', 'base': 'berlin52', 'k': 32, 'otimo': None, 'faixa': 'principal'},  # EUC_2D
    {'nome': 'berlin52_k38', 'base': 'berlin52', 'k': 38, 'otimo': None, 'faixa': 'principal'},  # EUC_2D
    # --- faixa limite (N em [45, 75]) ---
    {'nome': 'att48',     'otimo': 10628, 'faixa': 'limite'},     # ATT
    {'nome': 'berlin52',  'otimo': 7542,  'faixa': 'limite'},     # EUC_2D
    {'nome': 'st70',      'otimo': 675,   'faixa': 'limite'},     # EUC_2D
]


def _achar_arquivo(pasta, nome):
    for ext in ('.tsp', '.tsp.gz'):
        caminho = os.path.join(pasta, nome + ext)
        if os.path.exists(caminho):
            return caminho
    raise FileNotFoundError(f"{nome}.tsp não encontrado em {pasta}")


def carregar_instancias(pasta, nomes=None):
    """
    Devolve (lista de tuplas (nome, n, coords, distancias), dict nome -> ótimo conhecido),
    no formato que `rodar_experimento_bnb` aceita.
    """
    instancias, otimos = [], {}
    cache = {}
    for entrada in INSTANCIAS:
        nome = entrada['nome']
        if nomes is not None and nome not in nomes:
            continue
        base = entrada.get('base', nome)
        if base not in cache:
            cache[base] = ler_instancia_tsplib_completa(_achar_arquivo(pasta, base))
        d = cache[base]

        if 'k' in entrada:  # sub-instância: k primeiros pontos
            if d['tipo'] == 'EXPLICIT' or not d['coords_reais']:
                raise ValueError(f"{base}: sub-instância exige coordenadas reais")
            k = entrada['k']
            coords = {i: d['coords'][i] for i in range(k)}
            dist = calcular_matriz_distancias(coords, k, d['tipo'])
            instancias.append((nome, k, coords, dist))
        else:
            instancias.append((nome, d['n'], d['coords'], d['distancias']))
        otimos[nome] = entrada['otimo']
    return instancias, otimos


def validar(pasta):
    """Confere tipo, simetria e plausibilidade (qualquer rota >= ótimo conhecido)."""
    instancias, otimos = carregar_instancias(pasta)
    ok = True
    for nome, n, coords, dist in instancias:
        simetrica = all(dist[i][j] == dist[j][i] for i in range(n) for j in range(n))
        _, custo_nn = nearest_neighbor(dist, n)
        otimo = otimos[nome]
        if otimo is None:
            status = 'ótimo desconhecido (B&B calcula)'
        elif custo_nn < otimo:
            status = 'ERRO: heurística < ótimo -> tipo de distância errado?'
            ok = False
        else:
            status = f'NN/ótimo = {custo_nn / otimo:.2f}'
        if not simetrica:
            status += ' | ERRO: matriz assimétrica'
            ok = False
        print(f"{nome:<14} n={n:<3} coords={'sim' if coords else 'não':<4} NN={custo_nn:<7} {status}")
    print("\nOK" if ok else "\nHÁ PROBLEMAS")


if __name__ == '__main__':
    validar(sys.argv[1] if len(sys.argv) > 1 else 'data')
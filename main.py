"""
Script principal do TP2 — Caixeiro Viajante (TSP).

Lê instâncias da TSPLIB, executa Twice-around-the-tree e Christofides,
e exibe as métricas de tempo, memória e qualidade das rotas.

Uso:
    python main.py                    → usa instância embutida de teste
    python main.py data/ulysses16.tsp → usa uma instância da TSPLIB
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import ler_instancia_tsplib, calcular_matriz_distancias
from profiling import rodar_profiling, avaliar_qualidade


# Soluções ótimas conhecidas das instâncias da TSPLIB (EUC_2D)
# Fonte: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
OTIMOS_CONHECIDOS = {
    'ulysses16' : 6859,
    'ulysses22' : 7013,
    'gr24'      : 1272,
    'fri26'     : 937,
    'bays29'    : 2020,
    'dantzig42' : 699,
    'swiss42'   : 1273,
    'att48'     : 10628,
    'eil51'     : 426,
    'berlin52'  : 7542,
    'st70'      : 675,
    'eil76'     : 538,
}


def instancia_de_teste_embutida():
    """
    Instância pequena criada manualmente para demonstração rápida.
    5 vértices formando um pentagono regular aproximado.
    """
    import math
    n = 5
    # Vértices em um pentágono regular de raio 10
    coords = {}
    for i in range(n):
        angulo = 2 * math.pi * i / n
        coords[i] = (10 * math.cos(angulo), 10 * math.sin(angulo))
    return "teste_pentagono", n, coords


def main():
    # Lê instância da linha de comando ou usa a embutida
    if len(sys.argv) >= 2:
        caminho = sys.argv[1]
        print(f"Lendo instância: {caminho}")
        nome, n, coords = ler_instancia_tsplib(caminho)
    else:
        print("Nenhuma instância fornecida. Usando instância de teste embutida.")
        nome, n, coords = instancia_de_teste_embutida()

    print(f"Instância: {nome} | Vértices: {n}")

    # Monta a matriz de distâncias
    distancias = calcular_matriz_distancias(coords, n)

    # Busca solução ótima (se disponível)
    custo_otimo = OTIMOS_CONHECIDOS.get(nome.lower(), None)
    if custo_otimo:
        print(f"Ótimo conhecido: {custo_otimo}")
    else:
        print("Ótimo conhecido: NA")

    # Executa profiling de todas as subrotinas
    resultados = rodar_profiling(distancias, n, nome)

    # Avalia qualidade das rotas
    avaliacao = avaliar_qualidade(resultados, custo_otimo)

    # Exibe as rotas encontradas
    print("\n--- Rotas encontradas ---")
    rota_t = resultados['twice_around']['rota']
    rota_c = resultados['christofides']['rota']
    print(f"  Twice-around-tree : {rota_t}")
    print(f"  Christofides      : {rota_c}")


if __name__ == "__main__":
    main()

"""
Funções utilitárias para o TP2 - Caixeiro Viajante (TSP).

Leitura de instâncias no formato TSPLIB (EUC_2D) e cálculo de distâncias.
"""
import math


def ler_instancia_tsplib(caminho_arquivo):
    """
    Lê uma instância da TSPLIB no formato de texto (EUC_2D).

    Retorna:
        nome       : nome da instância (string)
        n          : número de vértices
        coordenadas: dicionário {índice: (x, y)}, índices começando em 0
    """
    coordenadas = {}
    nome = ""
    n = 0
    lendo_coords = False

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if linha.startswith("NAME"):
                nome = linha.split(":")[1].strip()
            elif linha.startswith("DIMENSION"):
                n = int(linha.split(":")[1].strip())
            elif linha == "NODE_COORD_SECTION":
                lendo_coords = True
            elif linha in ("EOF", ""):
                lendo_coords = False
            elif lendo_coords:
                partes = linha.split()
                # TSPLIB usa índices de 1 a N; convertemos para 0..N-1
                idx = int(partes[0]) - 1
                x = float(partes[1])
                y = float(partes[2])
                coordenadas[idx] = (x, y)

    return nome, n, coordenadas


def distancia_euc_2d(coord1, coord2):
    """
    Distância euclidiana 2D conforme a especificação da TSPLIB.
    Arredonda para o inteiro mais próximo (nint), como define o formato EUC_2D.

    Referência: TSPLIB95 documentation (Reinelt, 1991)
    """
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    return int(math.sqrt(xd * xd + yd * yd) + 0.5)


def calcular_matriz_distancias(coordenadas, n):
    """
    Monta a matriz de distâncias n×n entre todos os pares de vértices.

    distancias[i][j] = distância euclidiana entre o vértice i e o vértice j.
    A matriz é simétrica: distancias[i][j] == distancias[j][i].
    """
    distancias = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distancia_euc_2d(coordenadas[i], coordenadas[j])
            distancias[i][j] = d
            distancias[j][i] = d
    return distancias


def calcular_custo_rota(rota, distancias):
    """
    Calcula o custo total de uma rota (ciclo hamiltoniano).

    A rota é uma lista de vértices. O custo inclui a volta do último ao primeiro.
    """
    custo = 0
    n = len(rota)
    for i in range(n):
        u = rota[i]
        v = rota[(i + 1) % n]
        custo += distancias[u][v]
    return custo

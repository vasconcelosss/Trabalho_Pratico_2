import math

def ler_instancia_tsplib(caminho_arquivo):
    coordenadas = {}
    nome = ""
    n = 0
    edge_weight_type = ""
    edge_weight_format = ""
    lendo_coords = False
    lendo_pesos = False
    valores_pesos = []  # todos os números do EDGE_WEIGHT_SECTION em ordem

    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            linha = linha.strip()

            if linha.startswith("NAME"):
                nome = linha.split(":", 1)[1].strip()
            elif linha.startswith("DIMENSION"):
                n = int(linha.split(":", 1)[1].strip())
            elif linha.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = linha.split(":", 1)[1].strip()
            elif linha.startswith("EDGE_WEIGHT_FORMAT"):
                edge_weight_format = linha.split(":", 1)[1].strip()
            elif linha == "NODE_COORD_SECTION":
                lendo_coords = True
                lendo_pesos = False
            elif linha == "EDGE_WEIGHT_SECTION":
                lendo_coords = False
                lendo_pesos = True
            elif linha in ("EOF", "DISPLAY_DATA_SECTION", "NODE_COORD_SECTION"):
                lendo_coords = "DISPLAY_DATA_SECTION" not in linha
                lendo_pesos = False
            elif lendo_coords and linha:
                partes = linha.split()
                # TSPLIB usa índices de 1 a N; convertemos para 0..N-1
                idx = int(partes[0]) - 1
                x = float(partes[1])
                y = float(partes[2])
                coordenadas[idx] = (x, y)
            elif lendo_pesos and linha:
                for val in linha.split():
                    valores_pesos.append(int(val))

    # Monta a matriz explícita quando há seção de pesos
    distancias_predef = None
    if valores_pesos and n > 0:
        distancias_predef = [[0] * n for _ in range(n)]

        if edge_weight_format == "LOWER_DIAG_ROW":
            # Triângulo inferior + diagonal: linha i tem (i+1) valores
            idx = 0
            for i in range(n):
                for j in range(i + 1):
                    distancias_predef[i][j] = valores_pesos[idx]
                    distancias_predef[j][i] = valores_pesos[idx]
                    idx += 1

        elif edge_weight_format == "UPPER_DIAG_ROW":
            # Triângulo superior + diagonal: linha i tem (n-i) valores
            idx = 0
            for i in range(n):
                for j in range(i, n):
                    distancias_predef[i][j] = valores_pesos[idx]
                    distancias_predef[j][i] = valores_pesos[idx]
                    idx += 1

        elif edge_weight_format == "LOWER_ROW":
            # Triângulo inferior sem diagonal: linha i tem i valores (i de 1 a n-1)
            idx = 0
            for i in range(1, n):
                for j in range(i):
                    distancias_predef[i][j] = valores_pesos[idx]
                    distancias_predef[j][i] = valores_pesos[idx]
                    idx += 1

        elif edge_weight_format == "UPPER_ROW":
            # Triângulo superior sem diagonal
            idx = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    distancias_predef[i][j] = valores_pesos[idx]
                    distancias_predef[j][i] = valores_pesos[idx]
                    idx += 1

        elif edge_weight_format == "FULL_MATRIX":
            idx = 0
            for i in range(n):
                for j in range(n):
                    distancias_predef[i][j] = valores_pesos[idx]
                    idx += 1

        else:
            distancias_predef = None
            print(f"Aviso: formato '{edge_weight_format}' não suportado — instância ignorada.")

    return nome, n, coordenadas, distancias_predef


def distancia_euc_2d(coord1, coord2):
    xd = coord1[0] - coord2[0]
    yd = coord1[1] - coord2[1]
    return int(math.sqrt(xd * xd + yd * yd) + 0.5)


def calcular_matriz_distancias(coordenadas, n):
    distancias = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distancia_euc_2d(coordenadas[i], coordenadas[j])
            distancias[i][j] = d
            distancias[j][i] = d
    return distancias


def calcular_custo_rota(rota, distancias):
    custo = 0
    n = len(rota)
    for i in range(n):
        u = rota[i]
        v = rota[(i + 1) % n]
        custo += distancias[u][v]
    return custo

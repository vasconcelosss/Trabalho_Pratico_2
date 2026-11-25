import gzip
import math

PI_TSPLIB = 3.141592
RAIO_TERRA = 6378.388

TIPOS_COORD = ("EUC_2D", "CEIL_2D", "GEO", "ATT")
_SECOES_CONHECIDAS = {"NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DISPLAY_DATA_SECTION"}


# ---------------------------------------------------------------------------
# Funções de distância (TSPLIB95, seção 2.1)
# ---------------------------------------------------------------------------
def _nint(x):
    return int(x + 0.5)


def distancia_euc_2d(c1, c2):
    xd = c1[0] - c2[0]
    yd = c1[1] - c2[1]
    return _nint(math.sqrt(xd * xd + yd * yd))


def distancia_ceil_2d(c1, c2):
    xd = c1[0] - c2[0]
    yd = c1[1] - c2[1]
    return int(math.ceil(math.sqrt(xd * xd + yd * yd)))


def _geo_para_radianos(x):
    # x vem em DD.MM (graus e minutos), não em graus decimais
    graus = int(x)
    minutos = x - graus
    return PI_TSPLIB * (graus + 5.0 * minutos / 3.0) / 180.0


def distancia_geo(c1, c2):
    lat1, lon1 = _geo_para_radianos(c1[0]), _geo_para_radianos(c1[1])
    lat2, lon2 = _geo_para_radianos(c2[0]), _geo_para_radianos(c2[1])
    q1 = math.cos(lon1 - lon2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    arg = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    arg = max(-1.0, min(1.0, arg))  # evita erro de domínio por arredondamento
    return int(RAIO_TERRA * math.acos(arg) + 1.0)


def distancia_att(c1, c2):
    xd = c1[0] - c2[0]
    yd = c1[1] - c2[1]
    r = math.sqrt((xd * xd + yd * yd) / 10.0)
    t = _nint(r)
    return t + 1 if t < r else t


FUNCOES_DISTANCIA = {
    "EUC_2D": distancia_euc_2d,
    "CEIL_2D": distancia_ceil_2d,
    "GEO": distancia_geo,
    "ATT": distancia_att,
}


def calcular_matriz_distancias(coordenadas, n, tipo="EUC_2D"):
    if tipo not in FUNCOES_DISTANCIA:
        raise ValueError(f"Tipo de distância não suportado: '{tipo}'")
    f = FUNCOES_DISTANCIA[tipo]
    distancias = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = f(coordenadas[i], coordenadas[j])
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


# ---------------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------------
def _abrir(caminho):
    if caminho.endswith(".gz"):
        return gzip.open(caminho, "rt")
    return open(caminho, "r")


def _matriz_explicita(valores, n, formato):
    esperado = {
        "LOWER_DIAG_ROW": n * (n + 1) // 2,
        "UPPER_DIAG_ROW": n * (n + 1) // 2,
        "LOWER_ROW": n * (n - 1) // 2,
        "UPPER_ROW": n * (n - 1) // 2,
        "FULL_MATRIX": n * n,
    }
    if formato not in esperado:
        raise ValueError(f"EDGE_WEIGHT_FORMAT '{formato}' não suportado")
    if len(valores) != esperado[formato]:
        raise ValueError(
            f"{formato}: esperados {esperado[formato]} valores, lidos {len(valores)}"
        )

    d = [[0] * n for _ in range(n)]
    it = iter(valores)
    if formato == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(i + 1):
                d[i][j] = d[j][i] = next(it)
    elif formato == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                d[i][j] = d[j][i] = next(it)
    elif formato == "LOWER_ROW":
        for i in range(1, n):
            for j in range(i):
                d[i][j] = d[j][i] = next(it)
    elif formato == "UPPER_ROW":
        for i in range(n - 1):
            for j in range(i + 1, n):
                d[i][j] = d[j][i] = next(it)
    else:  # FULL_MATRIX
        for i in range(n):
            for j in range(n):
                d[i][j] = next(it)
    for i in range(n):
        d[i][i] = 0
    return d


def _numero(txt):
    v = float(txt)
    return int(v) if v == int(v) else v


def ler_instancia_tsplib_completa(caminho_arquivo):
    """
    Lê uma instância TSPLIB (.tsp ou .tsp.gz) e devolve um dicionário:
      nome, n, tipo (EDGE_WEIGHT_TYPE), coords {idx: (x, y)},
      coords_reais (False se vieram de DISPLAY_DATA, só para desenho),
      distancias (matriz n×n já calculada conforme o tipo).
    Levanta ValueError para tipos/formatos não suportados.
    """
    nome, n, tipo, formato = "", 0, "", ""
    coords_node, coords_display = {}, {}
    valores_pesos = []
    secao = None

    with _abrir(caminho_arquivo) as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            if linha == "EOF":
                break
            if linha.endswith("_SECTION"):
                secao = linha if linha in _SECOES_CONHECIDAS else "IGNORAR"
                continue
            if ":" in linha:  # cabeçalho "CHAVE : valor"
                chave, valor = linha.split(":", 1)
                chave, valor = chave.strip(), valor.strip()
                if chave == "NAME":
                    nome = valor
                elif chave == "DIMENSION":
                    n = int(valor)
                elif chave == "EDGE_WEIGHT_TYPE":
                    tipo = valor
                elif chave == "EDGE_WEIGHT_FORMAT":
                    formato = valor
                continue

            partes = linha.split()
            if secao == "NODE_COORD_SECTION":
                coords_node[int(partes[0]) - 1] = (float(partes[1]), float(partes[2]))
            elif secao == "DISPLAY_DATA_SECTION":
                coords_display[int(partes[0]) - 1] = (float(partes[1]), float(partes[2]))
            elif secao == "EDGE_WEIGHT_SECTION":
                valores_pesos.extend(_numero(p) for p in partes)

    if tipo in TIPOS_COORD:
        if len(coords_node) != n:
            raise ValueError(f"{nome}: {len(coords_node)} coordenadas lidas, DIMENSION={n}")
        coords, coords_reais = coords_node, True
        distancias = calcular_matriz_distancias(coords, n, tipo)
    elif tipo == "EXPLICIT":
        distancias = _matriz_explicita(valores_pesos, n, formato)
        coords = coords_display if len(coords_display) == n else {}
        coords_reais = False
    else:
        raise ValueError(f"{nome or caminho_arquivo}: EDGE_WEIGHT_TYPE '{tipo}' não suportado")

    return {
        "nome": nome, "n": n, "tipo": tipo, "coords": coords,
        "coords_reais": coords_reais, "distancias": distancias,
    }


def ler_instancia_tsplib(caminho_arquivo):
    """
    Interface antiga: (nome, n, coords, distancias).
    Agora o 4º elemento é SEMPRE a matriz completa (calculada conforme o
    EDGE_WEIGHT_TYPE), então quem chama não deve mais assumir EUC_2D.
    """
    d = ler_instancia_tsplib_completa(caminho_arquivo)
    return d["nome"], d["n"], d["coords"], d["distancias"]
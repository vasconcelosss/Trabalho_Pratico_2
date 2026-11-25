import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import (ler_instancia_tsplib_completa, calcular_matriz_distancias)
from profiling import rodar_profiling, avaliar_qualidade


# Soluções ótimas conhecidas das instâncias da TSPLIB (EUC_2D)
# Fonte: https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
OTIMOS_CONHECIDOS = {
    'burma14'   : 3323,
    'ulysses16' : 6859,
    'gr17'      : 2085,
    'gr21'      : 2707,
    'ulysses22' : 7013,
    'gr24'      : 1272,
    'fri26'     : 937,
    'bayg29'    : 1610,
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
    5 vértices formando um pentágono regular aproximado.
    """
    import math
    n = 5
    coords = {}
    for i in range(n):
        angulo = 2 * math.pi * i / n
        coords[i] = (10 * math.cos(angulo), 10 * math.sin(angulo))
    distancias = calcular_matriz_distancias(coords, n, "EUC_2D")
    return {"nome": "teste_pentagono", "n": n, "tipo": "EUC_2D",
            "coords": coords, "coords_reais": True, "distancias": distancias}


def rodar_bnb():
    """python main.py bnb [pasta=data] [timeout_s=1800]"""
    from instancias import carregar_instancias
    from experimento_bnb import rodar_experimento_bnb

    pasta = sys.argv[2] if len(sys.argv) > 2 else "data"
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 1800.0
    inst, otimos = carregar_instancias(pasta)
    rodar_experimento_bnb(inst, arquivo_saida="results/resultados_bnb.csv", timeout=timeout)


def main():
    # experimento do Branch-and-Bound com as 5 estratégias
    if len(sys.argv) >= 2 and sys.argv[1] == "bnb":
        rodar_bnb()
        return

    # le instância da linha de comando ou usa a embutida
    if len(sys.argv) >= 2:
        caminho = sys.argv[1]
        print(f"Lendo instância: {caminho}")
        inst = ler_instancia_tsplib_completa(caminho)
    else:
        print("Nenhuma instância fornecida. Usando instância de teste embutida.")
        inst = instancia_de_teste_embutida()

    nome, n, coords, distancias = inst["nome"], inst["n"], inst["coords"], inst["distancias"]
    print(f"Instância: {nome} | Vértices: {n}")
    print(f"Tipo de distância: {inst['tipo']}")

    custo_otimo = OTIMOS_CONHECIDOS.get(nome.lower(), None)
    if custo_otimo:
        print(f"Ótimo conhecido: {custo_otimo}")
    else:
        print("Ótimo conhecido: NA")

    resultados = rodar_profiling(distancias, n, nome)

    avaliacao = avaliar_qualidade(resultados, custo_otimo)

    print("\n--- Rotas encontradas ---")
    rota_t = resultados['twice_around']['rota']
    rota_c = resultados['christofides']['rota']
    print(f"  Twice-around-tree : {rota_t}")
    print(f"  Christofides      : {rota_c}")


if __name__ == "__main__":
    main()
    
    
# rodaria assim:
#python main.py                      # pentágono embutido
#python main.py data/gr24.tsp        # uma instância (como antes)
#python main.py bnb data 60          # B&B com timeout de 60 s (teste)
#python main.py bnb data             # B&B com 30 min
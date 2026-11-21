"""
Salva os resultados dos experimentos em CSV na pasta results/.
"""
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils import ler_instancia_tsplib, calcular_matriz_distancias
from profiling import rodar_profiling, avaliar_qualidade, calcular_gap


OTIMOS_CONHECIDOS = {
    'ulysses16'  : 6859,
    'ulysses22'  : 7013,
    'gr24'       : 1272,
    'fri26'      : 937,
    'bays29'     : 2020,
    'dantzig42'  : 699,
    'swiss42'    : 1273,
    'att48'      : 10628,
    'eil51'      : 426,
    'berlin52'   : 7542,
    'st70'       : 675,
    'eil76'      : 538,
}


def rodar_e_salvar(lista_arquivos_tsp, arquivo_saida="results/resultados.csv"):
    """
    Executa os algoritmos para cada instância e salva os resultados em CSV.
    """
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    colunas = [
        'instancia', 'n',
        'peso_mst',
        't_mst_ms', 'mem_mst_kb',
        't_matching_ms', 'mem_matching_kb',
        'custo_twice', 't_twice_ms', 'mem_twice_kb',
        'custo_christofides', 't_christofides_ms', 'mem_christofides_kb',
        'otimo', 'gap_twice_pct', 'gap_christofides_pct',
    ]

    with open(arquivo_saida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()

        for caminho in lista_arquivos_tsp:
            try:
                nome, n, coords = ler_instancia_tsplib(caminho)
                distancias = calcular_matriz_distancias(coords, n)
                custo_otimo = OTIMOS_CONHECIDOS.get(nome.lower(), None)

                res = rodar_profiling(distancias, n, nome)

                gap_t = calcular_gap(res['twice_around']['custo_rota'], custo_otimo)
                gap_c = calcular_gap(res['christofides']['custo_rota'], custo_otimo)

                linha = {
                    'instancia': nome,
                    'n': n,
                    'peso_mst': round(res['mst']['peso'], 2),
                    't_mst_ms': round(res['mst']['tempo_s'] * 1000, 4),
                    'mem_mst_kb': round(res['mst']['memoria_kb'], 2),
                    't_matching_ms': round(res['matching']['tempo_s'] * 1000, 4),
                    'mem_matching_kb': round(res['matching']['memoria_kb'], 2),
                    'custo_twice': round(res['twice_around']['custo_rota'], 2),
                    't_twice_ms': round(res['twice_around']['tempo_s'] * 1000, 4),
                    'mem_twice_kb': round(res['twice_around']['memoria_kb'], 2),
                    'custo_christofides': round(res['christofides']['custo_rota'], 2),
                    't_christofides_ms': round(res['christofides']['tempo_s'] * 1000, 4),
                    'mem_christofides_kb': round(res['christofides']['memoria_kb'], 2),
                    'otimo': custo_otimo if custo_otimo else 'NA',
                    'gap_twice_pct': round(gap_t, 4) if gap_t is not None else 'NA',
                    'gap_christofides_pct': round(gap_c, 4) if gap_c is not None else 'NA',
                }
                writer.writerow(linha)
                print(f"  Salvo: {nome}")

            except Exception as e:
                print(f"  ERRO em {caminho}: {e}")

    print(f"\nResultados salvos em: {arquivo_saida}")

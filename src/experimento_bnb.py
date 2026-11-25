import csv
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from utils import ler_instancia_tsplib
from branch_and_bound import executar_warm_starts


def rodar_experimento_bnb(lista_arquivos_tsp, arquivo_saida="results/resultados_bnb.csv", timeout=1800.0):
    os.makedirs(os.path.dirname(arquivo_saida), exist_ok=True)

    colunas = [
        'instancia', 'n', 'estrategia', 'status',
        'custo_inicial', 't_heuristica_ms', 'mem_heuristica_kb',
        'custo_otimo', 'melhor_custo_parcial',
        't_bnb_ms', 't_total_ms', 't_lower_bound_ms', 'mem_bnb_kb',
        'nos_expandidos', 'nos_podados', 'profundidade_max',
    ]

    with open(arquivo_saida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colunas)
        writer.writeheader()

        for item in lista_arquivos_tsp:
            # item pode ser um caminho de arquivo ou uma tupla (nome, n, coords, distancias)
            if isinstance(item, tuple):
                nome, n, coords, distancias = item
            else:
                nome, n, coords, distancias = ler_instancia_tsplib(item)

            print(f"\n=== Executando B&B + Warm-Starts: {nome} (n={n}) ===")
            resultados = executar_warm_starts(distancias, n, coords=coords, timeout=timeout)

            for estrategia, r in resultados.items():
                if r.get('status') == 'NA_SEM_COORDS':
                    continue

                linha = {
                    'instancia': nome,
                    'n': n,
                    'estrategia': estrategia,
                    'status': r['status'],
                    'custo_inicial': r['custo_inicial'],
                    't_heuristica_ms': round(r['t_heuristica_s'] * 1000, 4),
                    'mem_heuristica_kb': round(r['mem_heuristica_kb'], 2),
                    'custo_otimo': r['custo_otimo'] if r['custo_otimo'] is not None else 'NA',
                    'melhor_custo_parcial': r['melhor_custo_parcial'],
                    't_bnb_ms': round(r['t_bnb_s'] * 1000, 4) if r['t_bnb_s'] is not None else 'NA',
                    't_total_ms': round(r['t_total_s'] * 1000, 4) if r['t_total_s'] is not None else 'NA',
                    't_lower_bound_ms': round(r['t_lower_bound_s'] * 1000, 4),
                    'mem_bnb_kb': round(r['mem_bnb_kb'], 2),
                    'nos_expandidos': r['nos_expandidos'],
                    'nos_podados': r['nos_podados'],
                    'profundidade_max': r['profundidade_max'],
                }
                writer.writerow(linha)
                f.flush()
                print(
                    f"  {estrategia:<17} | {r['status']:<7} | "
                    f"exp={r['nos_expandidos']:<8} | pod={r['nos_podados']:<8} | "
                    f"t_bnb={linha['t_bnb_ms']} ms"
                )
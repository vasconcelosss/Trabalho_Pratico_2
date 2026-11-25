import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from medicao import medir as _medir
from mst import prim, peso_total_mst
from matching import emparelhamento_perfeito_minimo
from twice_around import twice_around_the_tree
from christofides import christofides


def medir(funcao, *args, repeticoes=5):
    """
    Executa uma função e retorna (resultado, tempo em segundos, pico de memória em bytes).
    Tempo = mediana de `repeticoes` execuções SEM tracemalloc; memória = passada
    separada com tracemalloc (ver medicao.py).
    """
    resultado, tempo_seg, pico_kb = _medir(funcao, *args, repeticoes=repeticoes)
    return resultado, tempo_seg, pico_kb * 1024


def rodar_profiling(distancias, n, nome_instancia="?"):
    """
    Roda todas as subrotinas e coleta métricas de tempo e memória.

    Retorna um dicionário com os resultados de cada subrotina.
    """
    print(f"\n--- Profiling: {nome_instancia} (n={n}) ---")

    resultados = {}

    # --- Subrotina: MST (Prim) ---
    (arestas_mst,), t_mst, mem_mst = medir(lambda d, k: (prim(d, k),), distancias, n)
    peso_mst = peso_total_mst(arestas_mst)
    resultados['mst'] = {
        'tempo_s': t_mst,
        'memoria_kb': mem_mst / 1024,
        'peso': peso_mst,
    }
    print(f"  MST        : {t_mst*1000:.3f} ms | {mem_mst/1024:.1f} KB | peso={peso_mst:.2f}")

    # --- Subrotina: Emparelhamento perfeito ---
    (arestas_m, peso_m), t_match, mem_match = medir(
        emparelhamento_perfeito_minimo, distancias, n, arestas_mst
    )
    resultados['matching'] = {
        'tempo_s': t_match,
        'memoria_kb': mem_match / 1024,
        'peso': peso_m,
    }
    print(f"  Matching   : {t_match*1000:.3f} ms | {mem_match/1024:.1f} KB | peso={peso_m:.2f}")

    # --- Subrotina: Twice-around-the-tree (completa) ---
    (rota_t, custo_t, _), t_twice, mem_twice = medir(
        twice_around_the_tree, distancias, n
    )
    resultados['twice_around'] = {
        'tempo_s': t_twice,
        'memoria_kb': mem_twice / 1024,
        'custo_rota': custo_t,
        'rota': rota_t,
    }
    print(f"  Twice-arnd : {t_twice*1000:.3f} ms | {mem_twice/1024:.1f} KB | custo={custo_t:.2f}")

    # --- Subrotina: Christofides (completa) ---
    (rota_c, custo_c, _, _), t_chris, mem_chris = medir(
        christofides, distancias, n
    )
    resultados['christofides'] = {
        'tempo_s': t_chris,
        'memoria_kb': mem_chris / 1024,
        'custo_rota': custo_c,
        'rota': rota_c,
    }
    print(f"  Christofid : {t_chris*1000:.3f} ms | {mem_chris/1024:.1f} KB | custo={custo_c:.2f}")

    return resultados


def calcular_gap(custo_heuristica, custo_otimo):
    """
    Calcula o gap percentual entre a heurística e a solução ótima.

    Gap(%) = (custo_heuristica - custo_otimo) / custo_otimo * 100

    Retorna None se o custo ótimo não for conhecido.
    """
    if custo_otimo is None or custo_otimo <= 0:
        return None
    return (custo_heuristica - custo_otimo) / custo_otimo * 100


def avaliar_qualidade(resultados, custo_otimo=None):
    """
    Avalia a qualidade das rotas comparando com a solução ótima (se disponível).

    Se não houver ótimo, registra NA.
    """
    print("\n--- Avaliação das rotas ---")

    custo_t = resultados['twice_around']['custo_rota']
    custo_c = resultados['christofides']['custo_rota']

    gap_t = calcular_gap(custo_t, custo_otimo)
    gap_c = calcular_gap(custo_c, custo_otimo)

    otimo_str = f"{custo_otimo:.2f}" if custo_otimo else "NA"
    gap_t_str = f"{gap_t:.2f}%" if gap_t is not None else "NA"
    gap_c_str = f"{gap_c:.2f}%" if gap_c is not None else "NA"

    print(f"  Ótimo             : {otimo_str}")
    print(f"  Twice-around-tree : {custo_t:.2f}  (gap: {gap_t_str})")
    print(f"  Christofides      : {custo_c:.2f}  (gap: {gap_c_str})")

    return {
        'custo_otimo': custo_otimo,
        'twice_around': {'custo': custo_t, 'gap': gap_t},
        'christofides': {'custo': custo_c, 'gap': gap_c},
    }
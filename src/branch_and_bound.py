import heapq
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from utils import calcular_custo_rota
from heuristics import cold_start, nearest_neighbor, convex_hull_insertion
from twice_around import twice_around_the_tree
from christofides import christofides
from medicao import medir_tempo, medir_memoria_kb


def _computar_menores_pesos(distancias, n):
    """
    Para cada vértice v retorna os dois menores pesos de arestas incidentes a ele.
    """
    m1 = [0.0] * n
    m2 = [0.0] * n
    for i in range(n):
        min1 = float('inf')
        min2 = float('inf')
        linha = distancias[i]
        for j in range(n):
            if i == j:
                continue
            d = linha[j]
            if d < min1:
                min2 = min1
                min1 = d
            elif d < min2:
                min2 = d
        m1[i] = min1
        m2[i] = min2
    return m1, m2


def _mst_subconjunto(distancias, vertices):
    """
    Algoritmo de Prim enxuto sobre um subconjunto de vértices (swap-and-pop).
    """
    k = len(vertices)
    if k <= 1:
        return 0.0

    u = vertices[0]
    linha_u = distancias[u]
    rest = vertices[1:]
    chave = [linha_u[v] for v in rest]
    peso_total = 0.0

    while rest:
        best_i = 0
        menor = chave[0]
        m = len(rest)
        for i in range(1, m):
            c = chave[i]
            if c < menor:
                menor = c
                best_i = i

        peso_total += menor
        u = rest[best_i]

        last_v = rest.pop()
        last_c = chave.pop()
        if best_i < len(rest):
            rest[best_i] = last_v
            chave[best_i] = last_c

        linha_u = distancias[u]
        for i, v in enumerate(rest):
            d = linha_u[v]
            if d < chave[i]:
                chave[i] = d

    return peso_total


def _potenciais_held_karp(distancias, n, custo_ub_ref=None, max_iter=80):
    """
    Calcula potenciais Lagrangeanos (Held-Karp / 1-árvore) na raiz via ascensão
    do subgradiente com passo adaptativo. Usa um UB de referência apenas para
    escalar o passo (sem alterar o upper bound real da busca).
    """
    if n <= 3:
        return [0.0] * n

    _, ub_chr = christofides(distancias, n)[:2]
    if custo_ub_ref is None or custo_ub_ref == float('inf'):
        ub_ref = ub_chr
    else:
        ub_ref = min(float(custo_ub_ref), float(ub_chr))

    pi = [0.0] * n
    melhor_pi = [0.0] * n
    melhor_lb = -float('inf')
    lam = 1.0
    sem_melhora = 0

    for _ in range(max_iter):
        chave = [float('inf')] * n
        pred = [-1] * n
        na_arvore = [False] * n
        chave[1] = 0.0
        peso_1tree = 0.0
        grau = [0] * n

        for _ in range(n - 1):
            u = -1
            menor = float('inf')
            for v in range(1, n):
                if not na_arvore[v] and chave[v] < menor:
                    menor = chave[v]
                    u = v
            na_arvore[u] = True
            peso_1tree += menor
            p = pred[u]
            if p != -1:
                grau[u] += 1
                grau[p] += 1

            linha_u = distancias[u]
            pi_u = pi[u]
            for v in range(1, n):
                if not na_arvore[v]:
                    w = linha_u[v] + pi_u + pi[v]
                    if w < chave[v]:
                        chave[v] = w
                        pred[v] = u

        min1_val = float('inf')
        min2_val = float('inf')
        min1_v = -1
        min2_v = -1
        linha_0 = distancias[0]
        pi_0 = pi[0]
        for v in range(1, n):
            w = linha_0[v] + pi_0 + pi[v]
            if w < min1_val:
                min2_val, min2_v = min1_val, min1_v
                min1_val, min1_v = w, v
            elif w < min2_val:
                min2_val, min2_v = w, v

        peso_1tree += min1_val + min2_val
        grau[0] = 2
        grau[min1_v] += 1
        grau[min2_v] += 1

        lb = peso_1tree - 2.0 * sum(pi)
        if lb > melhor_lb + 1e-9:
            melhor_lb = lb
            melhor_pi = pi[:]
            sem_melhora = 0
        else:
            sem_melhora += 1
            if sem_melhora >= 5:
                lam *= 0.5
                sem_melhora = 0

        norm2 = 0
        for v in range(1, n):
            g = grau[v] - 2
            norm2 += g * g
        if norm2 == 0:
            break

        gap = ub_ref - lb
        if gap <= 1e-9:
            break

        passo = lam * gap / norm2
        for v in range(1, n):
            pi[v] += passo * (grau[v] - 2)

    return melhor_pi


def branch_and_bound(distancias, n, rota_inicial=None, custo_inicial=None,
                     timeout=1800.0, estrategia='dfs', usar_held_karp=True):
    """
    Algoritmo exato Branch-and-Bound para o TSP simétrico.
    """
    t_inicio = time.perf_counter()

    if n <= 2:
        rota = list(range(n))
        custo = calcular_custo_rota(rota, distancias) if n > 0 else 0
        dt = time.perf_counter() - t_inicio
        return {
            'status': 'OK',
            'rota': rota,
            'custo': custo,
            'melhor_rota_parcial': rota,
            'melhor_custo_parcial': custo,
            'tempo_s': dt,
            'tempo_decorrido_s': dt,
            't_lower_bound_s': 0.0,
            'nos_expandidos': 1,
            'nos_podados': 0,
            'profundidade_max': n,
        }

    melhor_custo = float(custo_inicial) if custo_inicial is not None else float('inf')
    melhor_rota = list(rota_inicial) if rota_inicial is not None else None

    t_lb0 = time.perf_counter()
    if usar_held_karp:
        pi = _potenciais_held_karp(distancias, n, custo_ub_ref=melhor_custo)
        dist_lb = [
            [distancias[i][j] + pi[i] + pi[j] for j in range(n)]
            for i in range(n)
        ]
    else:
        pi = [0.0] * n
        dist_lb = distancias

    m1, m2 = _computar_menores_pesos(dist_lb, n)
    soma_m1_m2_inicial = sum(m1[v] + m2[v] for v in range(1, n))
    soma_pi_inicial = sum(pi[v] for v in range(1, n))
    pi_0 = pi[0]
    t_lower_bound_s = time.perf_counter() - t_lb0

    nos_expandidos = 0
    nos_podados = 0
    profundidade_max = 1
    abortado = False
    tol = 1e-9

    if estrategia == 'dfs':
        caminho = [0] * n
        visitados = [False] * n
        visitados[0] = True

        def dfs(prof, ultimo, custo_atual, soma_m1_m2, soma_pi):
            nonlocal melhor_custo, melhor_rota
            nonlocal nos_expandidos, nos_podados, profundidade_max
            nonlocal t_lower_bound_s, abortado

            if abortado:
                return

            if prof > profundidade_max:
                profundidade_max = prof

            if (nos_expandidos & 1023) == 0:
                if (time.perf_counter() - t_inicio) >= timeout:
                    abortado = True
                    return

            restantes = n - prof

            if restantes == 1:
                for v in range(1, n):
                    if not visitados[v]:
                        if prof >= 2 and v <= caminho[1]:
                            nos_podados += 1
                            return
                        custo_total = custo_atual + distancias[ultimo][v] + distancias[v][0]
                        if custo_total < melhor_custo:
                            melhor_custo = custo_total
                            caminho[prof] = v
                            melhor_rota = caminho[:]
                        break
                return

            nao_visitados = [v for v in range(1, n) if not visitados[v]]

            # quebra de simetria antes de reordenar (nao_visitados está em ordem crescente)
            if prof >= 2 and nao_visitados[-1] <= caminho[1]:
                nos_podados += 1
                return

            if melhor_custo < float('inf'):
                t_lb = time.perf_counter()
                linha_0 = dist_lb[0]
                linha_u = dist_lb[ultimo]
                min_0 = min(linha_0[v] for v in nao_visitados)
                min_u = min(linha_u[v] for v in nao_visitados)
                peso_mst = _mst_subconjunto(dist_lb, nao_visitados)
                t_lower_bound_s += time.perf_counter() - t_lb

                lb_mst = (
                    custo_atual + min_u + peso_mst + min_0
                    - pi_0 - pi[ultimo] - 2.0 * soma_pi
                )
                if lb_mst >= melhor_custo - tol:
                    nos_podados += 1
                    return

            nos_expandidos += 1

            # ordena pelo custo reduzido (segue as arestas da 1-árvore de Held-Karp)
            linha_lb_u = dist_lb[ultimo]
            nao_visitados.sort(key=lambda v: linha_lb_u[v])
            linha_real = distancias[ultimo]

            for v in nao_visitados:
                if abortado:
                    return

                if prof == 1 and v == n - 1:
                    nos_podados += 1
                    continue

                novo_custo = custo_atual + linha_real[v]
                if novo_custo >= melhor_custo - tol:
                    nos_podados += 1
                    continue

                nova_soma = soma_m1_m2 - (m1[v] + m2[v])
                nova_soma_pi = soma_pi - pi[v]

                if restantes - 1 >= 2 and melhor_custo < float('inf'):
                    lb_rapido = (
                        novo_custo + 0.5 * (m1[0] + m1[v] + nova_soma)
                        - pi_0 - pi[v] - 2.0 * nova_soma_pi
                    )
                    if lb_rapido >= melhor_custo - tol:
                        nos_podados += 1
                        continue

                visitados[v] = True
                caminho[prof] = v
                dfs(prof + 1, v, novo_custo, nova_soma, nova_soma_pi)
                visitados[v] = False

        dfs(1, 0, 0.0, soma_m1_m2_inicial, soma_pi_inicial)

    elif estrategia == 'best_first':
        seq = 0
        heap = [(0.0, -1, seq, (0,), 0.0, soma_m1_m2_inicial, soma_pi_inicial)]

        while heap:
            if (nos_expandidos & 1023) == 0:
                if (time.perf_counter() - t_inicio) >= timeout:
                    abortado = True
                    break

            lb, neg_prof, _, caminho_atual, custo_atual, soma_m1_m2, soma_pi = heapq.heappop(heap)
            prof = -neg_prof

            if lb >= melhor_custo - tol:
                nos_podados += 1
                continue

            if prof > profundidade_max:
                profundidade_max = prof

            ultimo = caminho_atual[-1]
            visitados_set = set(caminho_atual)
            nao_visitados = [v for v in range(1, n) if v not in visitados_set]
            restantes = len(nao_visitados)

            if prof >= 2 and nao_visitados[-1] <= caminho_atual[1]:
                nos_podados += 1
                continue

            if restantes == 1:
                v = nao_visitados[0]
                if prof >= 2 and v <= caminho_atual[1]:
                    nos_podados += 1
                    continue
                custo_total = custo_atual + distancias[ultimo][v] + distancias[v][0]
                if custo_total < melhor_custo:
                    melhor_custo = custo_total
                    melhor_rota = list(caminho_atual) + [v]
                continue

            nos_expandidos += 1
            linha_real = distancias[ultimo]
            linha_0 = dist_lb[0]

            for v in nao_visitados:
                if prof == 1 and v == n - 1:
                    nos_podados += 1
                    continue

                novo_custo = custo_atual + linha_real[v]
                if novo_custo >= melhor_custo - tol:
                    nos_podados += 1
                    continue

                nova_soma = soma_m1_m2 - (m1[v] + m2[v])
                nova_soma_pi = soma_pi - pi[v]
                novo_rest = restantes - 1

                if novo_rest == 1:
                    u_rest = next(x for x in nao_visitados if x != v)
                    v1 = v if prof == 1 else caminho_atual[1]
                    if u_rest <= v1:
                        nos_podados += 1
                        continue
                    custo_total = novo_custo + distancias[v][u_rest] + distancias[u_rest][0]
                    if custo_total < melhor_custo:
                        melhor_custo = custo_total
                        melhor_rota = list(caminho_atual) + [v, u_rest]
                    continue

                lb_rapido = (
                    novo_custo + 0.5 * (m1[0] + m1[v] + nova_soma)
                    - pi_0 - pi[v] - 2.0 * nova_soma_pi
                )
                if lb_rapido >= melhor_custo - tol:
                    nos_podados += 1
                    continue

                t_lb = time.perf_counter()
                sub = [x for x in nao_visitados if x != v]
                min_0 = min(linha_0[x] for x in sub)
                min_v = min(dist_lb[v][x] for x in sub)
                peso_mst = _mst_subconjunto(dist_lb, sub)
                t_lower_bound_s += time.perf_counter() - t_lb

                lb_mst = (
                    novo_custo + min_v + peso_mst + min_0
                    - pi_0 - pi[v] - 2.0 * nova_soma_pi
                )
                lb_filho = lb_mst if lb_mst > lb_rapido else lb_rapido

                if lb_filho >= melhor_custo - tol:
                    nos_podados += 1
                    continue

                seq += 1
                heapq.heappush(
                    heap,
                    (lb_filho, -(prof + 1), seq, caminho_atual + (v,),
                     novo_custo, nova_soma, nova_soma_pi),
                )
    else:
        raise ValueError(f"Estratégia desconhecida: '{estrategia}'. Use 'dfs' ou 'best_first'.")

    tempo_decorrido = time.perf_counter() - t_inicio
    status = 'TIMEOUT' if abortado else 'OK'

    return {
        'status': status,
        'rota': None if abortado else melhor_rota,
        'custo': None if abortado else melhor_custo,
        'melhor_rota_parcial': melhor_rota,
        'melhor_custo_parcial': melhor_custo,
        'tempo_s': None if abortado else tempo_decorrido,
        'tempo_decorrido_s': tempo_decorrido,
        't_lower_bound_s': t_lower_bound_s,
        'nos_expandidos': nos_expandidos,
        'nos_podados': nos_podados,
        'profundidade_max': profundidade_max,
    }


def executar_warm_starts(distancias, n, coords=None, timeout=1800.0, estrategia='dfs',
                         repeticoes_heuristica=5, timeout_mem=None):
    """
    Executa as 5 estratégias de inicialização (warm-start) e compara o impacto na árvore do b&b.
    """
    if timeout_mem is None:
        timeout_mem = 5.0 if estrategia == 'dfs' else 60.0

    estrategias = [
        ('cold_start', lambda: cold_start(distancias, n)),
        ('nearest_neighbor', lambda: nearest_neighbor(distancias, n)),
        ('convex_hull', lambda: convex_hull_insertion(distancias, n, coords) if coords else (None, None)),
        ('twice_around', lambda: twice_around_the_tree(distancias, n)[:2]),
        ('christofides', lambda: christofides(distancias, n)[:2]),
    ]

    resultados = {}

    for nome, gerador in estrategias:
        if nome == 'convex_hull' and (not coords or len(coords) < n):
            resultados[nome] = {'status': 'NA_SEM_COORDS'}
            continue

        (rota_init, custo_init), t_heur_s = medir_tempo(
            gerador, repeticoes=repeticoes_heuristica
        )
        _, mem_heur_kb = medir_memoria_kb(gerador)

        params_bnb = dict(
            rota_inicial=rota_init,
            custo_inicial=custo_init,
            estrategia=estrategia,
        )
        res_bnb, _ = medir_tempo(
            branch_and_bound, distancias, n, timeout=timeout, repeticoes=1, **params_bnb
        )
        _, mem_bnb_kb = medir_memoria_kb(
            branch_and_bound, distancias, n, timeout=min(timeout, timeout_mem), **params_bnb
        )

        tempo_total_s = (
            None if res_bnb['tempo_s'] is None else (t_heur_s + res_bnb['tempo_s'])
        )

        resultados[nome] = {
            'status': res_bnb['status'],
            'custo_inicial': custo_init,
            't_heuristica_s': t_heur_s,
            'mem_heuristica_kb': mem_heur_kb,
            'custo_otimo': res_bnb['custo'],
            'melhor_custo_parcial': res_bnb['melhor_custo_parcial'],
            't_bnb_s': res_bnb['tempo_s'],
            't_total_s': tempo_total_s,
            't_decorrido_s': res_bnb['tempo_decorrido_s'],
            't_lower_bound_s': res_bnb['t_lower_bound_s'],
            'mem_bnb_kb': mem_bnb_kb,
            'nos_expandidos': res_bnb['nos_expandidos'],
            'nos_podados': res_bnb['nos_podados'],
            'profundidade_max': res_bnb['profundidade_max'],
            'rota_otima': res_bnb['rota'],
        }

    return resultados
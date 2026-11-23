"""
Teste de Estresse — TP2 Algoritmos 2 (DCC207 / UFMG)

Executa os algoritmos do TP em instâncias de tamanho crescente e grava os
resultados em CSV e texto durante a execução (não só ao final).

COMO USAR:
  Modo rápido (n até 100):
    python -m tests.stress_test --mode quick

  Modo completo:
    python -m tests.stress_test --algorithm all --min-n 10 --max-n 2000 --step 25 --timeout 120 --repetitions 3

  Só Christofides:
    python -m tests.stress_test --algorithm christofides --min-n 10 --max-n 500 --timeout 120

  Só Matching:
    python -m tests.stress_test --algorithm matching --min-n 10 --max-n 300 --timeout 120

  Verificar se os imports estão corretos (sem rodar testes):
    python -m tests.stress_test --validate

Pressione CTRL+C para interromper com segurança — resultados já salvos são preservados.
"""

import sys
import os
import math
import time
import random
import csv
import statistics
import argparse
import signal
import queue
import multiprocessing

# Permite rodar como módulo (python -m tests.stress_test) a partir da raiz do projeto.
# Usa abspath para garantir caminho absoluto tanto no processo pai quanto nos filhos
# (multiprocessing.spawn recria o módulo em cada processo filho).
_SRC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from mst import prim, peso_total_mst, vertices_grau_impar, adjacencia_mst
    from twice_around import twice_around_the_tree
    from matching import emparelhamento_perfeito_minimo
    from christofides import christofides
    from utils import distancia_euc_2d, calcular_custo_rota
except ImportError as e:
    print(f"ERRO: Não foi possível importar módulos do projeto: {e}")
    print("Execute a partir da raiz do projeto: python -m tests.stress_test")
    sys.exit(1)

# psutil: monitora RSS do processo filho a partir do pai
try:
    import psutil
    PSUTIL_DISPONIVEL = True
except ImportError:
    PSUTIL_DISPONIVEL = False


# ───────────────────────────────────────────────────────────
# CONSTANTES
# ───────────────────────────────────────────────────────────

TODOS_ALGORITMOS = ['mst', 'twice', 'matching', 'christofides']

TODAS_DISTRIBUICOES = [
    'uniform',       # Pontos uniformes aleatórios
    'circle',        # Pontos igualmente espaçados em círculo
    'line',          # Pontos em linha reta
    'elongated',     # Retângulo muito alongado (20:1)
    'clusters',      # 4 clusters com pontos densos
    'convex_few',    # Poucos pontos no fecho convexo (gaussiana centrada)
    'convex_many',   # Muitos pontos no fecho convexo (círculo com ruído)
    'irregular',     # Distribuição irregular: mistura de uniforme + cluster
]

DIR_RESULTADOS = "results/stress_test"
ARQ_CSV        = os.path.join(DIR_RESULTADOS, "resultados.csv")
ARQ_RESUMO     = os.path.join(DIR_RESULTADOS, "resumo.txt")
DIR_GRAFICOS   = os.path.join(DIR_RESULTADOS, "graficos")

COLUNAS_CSV = [
    'n', 'distribuicao', 'semente', 'algoritmo', 'repeticao',
    't_total_ms',         # tempo do algoritmo (Blossom para matching, total para os demais)
    't_mst_ms',           # tempo da MST dentro do matching (0 para todos os outros algoritmos)
    't_matriz_ms',        # tempo de construção da matriz de distâncias
    'mem_rss_pico_mb',    # pico de RSS do processo filho via psutil (-1 se indisponível)
    'n_vertices_impares', 'peso_mst', 'custo_rota', 'rota_valida', 'status',
]

# Flag global para sinalizar interrupção segura (CTRL+C)
_interrompido = False


# ───────────────────────────────────────────────────────────
# WORKER — executa no processo filho
#
# DEVE ficar no nível do módulo (top-level) para que
# multiprocessing.spawn consiga localizá-lo no Windows.
# O subprocess importa tests.stress_test ao bootstrapar,
# portanto as funções do projeto (prim, christofides, etc.)
# já estão disponíveis como globais do módulo.
# ───────────────────────────────────────────────────────────

def _worker_loop(q_entrada, q_saida):
    """
    Loop do processo filho. Importa os algoritmos UMA VEZ ao subir e fica
    aguardando pedidos. Encerra ao receber None em q_entrada.

    Para 'matching': mede Prim e Blossom SEPARADAMENTE e retorna ambos os tempos.
    """
    import time

    while True:
        try:
            pedido = q_entrada.get(timeout=30)
        except Exception:
            # queue.Empty após timeout — continua aguardando
            continue

        if pedido is None:
            break

        alg_nome, dist, n = pedido
        try:
            if alg_nome == 'mst':
                t0  = time.perf_counter()
                res = prim(dist, n)
                q_saida.put({'status': 'OK', 'resultado': res,
                             't_total_ms': (time.perf_counter() - t0) * 1000,
                             't_mst_ms': 0.0})

            elif alg_nome == 'twice':
                t0  = time.perf_counter()
                res = twice_around_the_tree(dist, n)
                q_saida.put({'status': 'OK', 'resultado': res,
                             't_total_ms': (time.perf_counter() - t0) * 1000,
                             't_mst_ms': 0.0})

            elif alg_nome == 'matching':
                # Mede MST (Prim) e Matching (Blossom) em blocos separados
                t_mst0      = time.perf_counter()
                arestas_mst = prim(dist, n)
                t_mst_ms    = (time.perf_counter() - t_mst0) * 1000

                t_bl0        = time.perf_counter()
                match_res    = emparelhamento_perfeito_minimo(dist, n, arestas_mst)
                t_blossom_ms = (time.perf_counter() - t_bl0) * 1000

                q_saida.put({'status': 'OK',
                             'resultado': (match_res, arestas_mst),
                             't_total_ms': t_blossom_ms,   # só o Blossom
                             't_mst_ms':   t_mst_ms})       # só a MST

            elif alg_nome == 'christofides':
                t0  = time.perf_counter()
                res = christofides(dist, n)
                q_saida.put({'status': 'OK', 'resultado': res,
                             't_total_ms': (time.perf_counter() - t0) * 1000,
                             't_mst_ms': 0.0})

            else:
                q_saida.put({'status': f'ERRO: algoritmo desconhecido "{alg_nome}"',
                             'resultado': None, 't_total_ms': 0.0, 't_mst_ms': 0.0})

        except Exception as exc:
            q_saida.put({'status': f'ERRO: {exc}', 'resultado': None,
                         't_total_ms': 0.0, 't_mst_ms': 0.0})


# ───────────────────────────────────────────────────────────
# GERENCIADOR DO PROCESSO FILHO
# ───────────────────────────────────────────────────────────

class GerenciadorProcesso:
    """
    Mantém um processo filho persistente que executa os algoritmos.

    Por que multiprocessing em vez de threads:
    - terminate() / kill() encerram o processo de verdade, incluindo extensões C
      (como o Blossom do NetworkX que usa C por baixo do Python).
    - psutil monitora o RSS do filho a cada 100 ms e mata se exceder o limite.
    - Ao pressionar CTRL+C, o processo filho (daemon) é encerrado junto com o pai.

    Custo inicial: ~200–500 ms no Windows (spawn + importar NetworkX e módulos do TP).
    Testes seguintes reutilizam o mesmo processo.
    Após cada timeout, o processo é reiniciado (custo de spawn pago novamente).
    """

    def __init__(self):
        self._processo = None
        self._q_in     = None
        self._q_out    = None
        self._iniciar()

    def _iniciar(self):
        ctx = multiprocessing.get_context('spawn')
        self._q_in  = ctx.Queue()
        self._q_out = ctx.Queue()
        self._processo = ctx.Process(
            target=_worker_loop,
            args=(self._q_in, self._q_out),
            daemon=True,
        )
        self._processo.start()

    def _matar_e_reiniciar(self):
        """Encerra o processo filho (kill real) e cria um substituto."""
        if self._processo and self._processo.is_alive():
            self._processo.terminate()
            self._processo.join(3)
            if self._processo.is_alive():
                self._processo.kill()
                self._processo.join(1)
        self._iniciar()

    def executar(self, alg_nome, dist, n, timeout, max_memoria_mb=None):
        """
        Envia pedido ao processo filho e aguarda resultado.
        Monitora:
          - Timeout real: após expirar, chama p.terminate() + p.kill() (encerra C extensions).
          - Memória RSS: via psutil a cada 100 ms; se exceder o limite, encerra o processo.
          - CTRL+C: se _interrompido for True, encerra imediatamente.

        Retorna dict: {status, resultado, t_total_ms, t_mst_ms, mem_rss_pico_mb}
        """
        global _interrompido

        self._q_in.put((alg_nome, dist, n))
        inicio      = time.perf_counter()
        rss_pico_mb = 0.0
        pid_filho   = self._processo.pid

        while True:
            if _interrompido:
                self._matar_e_reiniciar()
                return {'status': 'INTERROMPIDO', 'resultado': None,
                        't_total_ms': 0.0, 't_mst_ms': 0.0,
                        'mem_rss_pico_mb': rss_pico_mb}

            decorrido = time.perf_counter() - inicio

            if decorrido >= timeout:
                self._matar_e_reiniciar()
                return {'status': 'TIMEOUT', 'resultado': None,
                        't_total_ms': round(decorrido * 1000, 2),
                        't_mst_ms': 0.0, 'mem_rss_pico_mb': rss_pico_mb}

            if PSUTIL_DISPONIVEL:
                try:
                    rss_mb = psutil.Process(pid_filho).memory_info().rss / (1024 * 1024)
                    if rss_mb > rss_pico_mb:
                        rss_pico_mb = rss_mb
                    if max_memoria_mb and rss_mb > max_memoria_mb:
                        self._matar_e_reiniciar()
                        return {'status': 'MEMORIA_EXCEDIDA', 'resultado': None,
                                't_total_ms': round(decorrido * 1000, 2),
                                't_mst_ms': 0.0, 'mem_rss_pico_mb': rss_pico_mb}
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            try:
                dados = self._q_out.get(timeout=0.1)
                dados.setdefault('mem_rss_pico_mb', rss_pico_mb)
                if dados['mem_rss_pico_mb'] < rss_pico_mb:
                    dados['mem_rss_pico_mb'] = rss_pico_mb
                return dados
            except queue.Empty:
                if not self._processo.is_alive():
                    exitcode = self._processo.exitcode
                    self._matar_e_reiniciar()
                    return {
                        'status': f'ERRO: processo filho encerrou (exit={exitcode})',
                        'resultado': None, 't_total_ms': round(decorrido * 1000, 2),
                        't_mst_ms': 0.0, 'mem_rss_pico_mb': rss_pico_mb,
                    }
            except KeyboardInterrupt:
                _interrompido = True
                self._matar_e_reiniciar()
                return {'status': 'INTERROMPIDO', 'resultado': None,
                        't_total_ms': 0.0, 't_mst_ms': 0.0,
                        'mem_rss_pico_mb': rss_pico_mb}

    def fechar(self):
        """Encerra o processo filho de forma limpa."""
        if self._processo and self._processo.is_alive():
            try:
                self._q_in.put(None)
                self._processo.join(3)
            except Exception:
                pass
            if self._processo and self._processo.is_alive():
                self._processo.terminate()
                self._processo.join(2)


# ───────────────────────────────────────────────────────────
# GERAÇÃO DE INSTÂNCIAS
# ───────────────────────────────────────────────────────────

def gerar_instancia(tipo, n, seed=42):
    """
    Gera coordenadas 2D para uma instância TSP artificial.
    Retorna dict {i: (x, y)} com n pontos.

    Tipos disponíveis e suas características para o TSP:
    - uniform     : distribuição mais comum em benchmarks aleatórios
    - circle      : MST tem apenas 2 vértices de grau ímpar → matching barato
    - line        : adversarial para Twice-around (DFS pode seguir caminho subótimo)
    - elongated   : razão de aspecto alta, testa geometrias não-quadradas
    - clusters    : simula casos reais com grupos geográficos
    - convex_few  : maioria dos pontos no interior → hull pequeno
    - convex_many : maioria dos pontos na borda → hull grande
    - irregular   : mistura intencionalmente não-uniforme
    """
    rng = random.Random(seed)

    if tipo == 'uniform':
        return {i: (rng.uniform(0, 1000), rng.uniform(0, 1000)) for i in range(n)}

    elif tipo == 'circle':
        return {i: (500 + 400 * math.cos(2 * math.pi * i / n),
                    500 + 400 * math.sin(2 * math.pi * i / n)) for i in range(n)}

    elif tipo == 'line':
        return {i: (i * 1000.0 / max(n - 1, 1), 0.0) for i in range(n)}

    elif tipo == 'elongated':
        return {i: (rng.uniform(0, 2000), rng.uniform(0, 100)) for i in range(n)}

    elif tipo == 'clusters':
        centros = [(200, 200), (800, 200), (200, 800), (800, 800)]
        return {i: (centros[i % 4][0] + rng.uniform(-80, 80),
                    centros[i % 4][1] + rng.uniform(-80, 80)) for i in range(n)}

    elif tipo == 'convex_few':
        return {i: (max(0.0, min(1000.0, rng.gauss(500, 120))),
                    max(0.0, min(1000.0, rng.gauss(500, 120)))) for i in range(n)}

    elif tipo == 'convex_many':
        return {i: (500 + (400 + rng.uniform(-8, 8)) * math.cos(2 * math.pi * rng.random()),
                    500 + (400 + rng.uniform(-8, 8)) * math.sin(2 * math.pi * rng.random()))
                for i in range(n)}

    elif tipo == 'irregular':
        coords = {}
        for i in range(n // 2):
            coords[i] = (rng.uniform(0, 1000), rng.uniform(0, 1000))
        cx = rng.uniform(200, 800)
        cy = rng.uniform(200, 800)
        for i in range(n // 2, n):
            coords[i] = (cx + rng.gauss(0, 25), cy + rng.gauss(0, 25))
        return coords

    else:
        raise ValueError(f"Distribuição desconhecida: '{tipo}'")


# ───────────────────────────────────────────────────────────
# MATRIZ DE DISTÂNCIAS
# ───────────────────────────────────────────────────────────

def montar_matriz(coords, n):
    """Constrói a matriz n×n de distâncias euclidianas."""
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distancia_euc_2d(coords[i], coords[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ───────────────────────────────────────────────────────────
# VALIDAÇÃO DE ROTA
# ───────────────────────────────────────────────────────────

def validar_rota(rota, n, dist):
    """Verifica se a rota é um ciclo hamiltoniano válido."""
    if len(rota) != n:
        return False
    if set(rota) != set(range(n)):
        return False
    if len(set(rota)) != n:
        return False
    return True


# ───────────────────────────────────────────────────────────
# WRAPPER POR ALGORITMO
# ───────────────────────────────────────────────────────────

def executar_algoritmo(nome_alg, dist, n, gerenciador, timeout, max_memoria_mb=None):
    """
    Executa o algoritmo via GerenciadorProcesso e retorna dicionário de métricas.
    n_vertices_impares e peso_mst são preenchidos quando disponíveis.

    Para 'matching', t_total_ms é o tempo do Blossom (C extension do NetworkX) e
    t_mst_ms é o tempo da construção da MST (Prim) feita previamente.
    """
    metricas = {
        't_total_ms':     0.0,
        't_mst_ms':       0.0,
        'mem_rss_pico_mb': -1.0,
        'n_vertices_impares': -1,
        'peso_mst':       0.0,
        'custo_rota':     0.0,
        'rota_valida':    False,
        'status':         'OK',
    }

    dados = gerenciador.executar(nome_alg, dist, n, timeout, max_memoria_mb)
    metricas['t_total_ms']      = dados.get('t_total_ms', 0.0)
    metricas['t_mst_ms']        = dados.get('t_mst_ms', 0.0)
    metricas['mem_rss_pico_mb'] = dados.get('mem_rss_pico_mb', -1.0)
    metricas['status']          = dados.get('status', 'ERRO')

    if dados['status'] != 'OK':
        return metricas

    resultado = dados['resultado']

    if nome_alg == 'mst':
        arestas = resultado
        metricas['n_vertices_impares'] = len(vertices_grau_impar(arestas, n))
        metricas['peso_mst']           = peso_total_mst(arestas)

    elif nome_alg == 'twice':
        rota, custo, arestas_mst = resultado
        metricas['custo_rota']         = custo
        metricas['rota_valida']        = validar_rota(rota, n, dist)
        metricas['n_vertices_impares'] = len(vertices_grau_impar(arestas_mst, n))
        metricas['peso_mst']           = peso_total_mst(arestas_mst)

    elif nome_alg == 'matching':
        # resultado = ((arestas_match, peso_match), arestas_mst)
        (arestas_match, peso_match), arestas_mst = resultado
        metricas['n_vertices_impares'] = len(vertices_grau_impar(arestas_mst, n))
        metricas['peso_mst']           = peso_match   # peso do matching, não da MST

    elif nome_alg == 'christofides':
        rota, custo, arestas_mst, peso_match = resultado
        metricas['custo_rota']         = custo
        metricas['rota_valida']        = validar_rota(rota, n, dist)
        metricas['n_vertices_impares'] = len(vertices_grau_impar(arestas_mst, n))
        metricas['peso_mst']           = peso_total_mst(arestas_mst)

    return metricas


# ───────────────────────────────────────────────────────────
# SALVAMENTO INCREMENTAL
# ───────────────────────────────────────────────────────────

class SalvadorCSV:
    """Mantém o arquivo CSV aberto e grava/faz flush a cada linha."""

    def __init__(self, caminho):
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        self._f   = open(caminho, 'w', newline='', encoding='utf-8')
        self._w   = csv.DictWriter(self._f, fieldnames=COLUNAS_CSV, extrasaction='ignore')
        self._w.writeheader()
        self._f.flush()

    def gravar(self, linha):
        self._w.writerow(linha)
        self._f.flush()

    def fechar(self):
        self._f.close()


def atualizar_resumo(resultados_por_n, algoritmos, caminho=ARQ_RESUMO):
    """
    Reescreve resumo.txt com as estatísticas acumuladas até o momento.
    Chamado após cada N ser concluído.
    """
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write("RESUMO DO STRESS TEST — TP2 ALGORITMOS 2\n")
        f.write("=" * 55 + "\n")
        f.write(f"Atualizado em: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"{'n':>6}  {'Algoritmo':<15}  {'T medio (ms)':>14}  {'Status'}\n")
        f.write("-" * 55 + "\n")

        for n in sorted(resultados_por_n.keys()):
            for alg in algoritmos:
                linhas = [r for r in resultados_por_n[n] if r['algoritmo'] == alg]
                ok     = [r for r in linhas if r['status'] == 'OK']
                if ok:
                    media = statistics.mean(r['t_total_ms'] for r in ok)
                    f.write(f"{n:>6}  {alg:<15}  {media:>14.3f}  OK ({len(ok)} reps)\n")
                elif linhas:
                    ultimo = linhas[-1]['status']
                    f.write(f"{n:>6}  {alg:<15}  {'---':>14}  {ultimo}\n")

        f.write("\n" + "=" * 55 + "\n")
        f.write(f"Resultados completos: {ARQ_CSV}\n")
        f.write(f"Graficos: {DIR_GRAFICOS}/\n")


# ───────────────────────────────────────────────────────────
# GRÁFICOS
# ───────────────────────────────────────────────────────────

def gerar_graficos(resultados_por_n, algoritmos):
    """
    Gera gráficos PNG de tempo × n.
    Chamado ao final ou quando CTRL+C é pressionado.
    Falha silenciosamente se matplotlib não estiver instalado.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [aviso] matplotlib nao encontrado -- graficos nao gerados.")
        print("  Para instalar: pip install matplotlib")
        return

    os.makedirs(DIR_GRAFICOS, exist_ok=True)

    cores = {
        'mst': 'steelblue',
        'twice': 'seagreen',
        'matching': 'darkorange',
        'christofides': 'crimson',
    }
    dados = {alg: {} for alg in algoritmos}
    for n, linhas in resultados_por_n.items():
        for alg in algoritmos:
            ok = [r['t_total_ms'] for r in linhas if r['algoritmo'] == alg and r['status'] == 'OK']
            if ok:
                dados[alg][n] = statistics.mean(ok)

    if not any(dados[alg] for alg in algoritmos):
        print("  Sem dados suficientes para gerar graficos.")
        return

    # Gráfico 1: todos os algoritmos (escala linear + log)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stress Test -- Tempo de Execucao", fontsize=13, fontweight='bold')

    for alg in algoritmos:
        if not dados[alg]:
            continue
        ns     = sorted(dados[alg].keys())
        tempos = [dados[alg][n] for n in ns]
        cor    = cores.get(alg, 'gray')
        axes[0].plot(ns, tempos, 'o-', label=alg, color=cor, linewidth=1.5, markersize=4)
        axes[1].plot(ns, tempos, 'o-', label=alg, color=cor, linewidth=1.5, markersize=4)

    for ax, titulo, logy in zip(axes,
                                 ["Escala linear", "Escala log (eixo Y)"],
                                 [False, True]):
        ax.set_xlabel("n (vertices)")
        ax.set_ylabel("Tempo medio (ms)")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale('log')

    plt.tight_layout()
    arq = os.path.join(DIR_GRAFICOS, "tempo_todos.png")
    plt.savefig(arq, dpi=120)
    plt.close()
    print(f"  Salvo: {arq}")

    # Gráfico 2: separado por grupo de velocidade
    rapidos = {alg: v for alg, v in dados.items() if alg in ('mst', 'twice') and v}
    lentos  = {alg: v for alg, v in dados.items() if alg in ('matching', 'christofides') and v}

    if rapidos or lentos:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Stress Test -- MST/Twice vs Matching/Christofides", fontsize=12)

        for grupo, ax, titulo in [
            (rapidos, axes[0], "MST e Twice-around (rapidos)"),
            (lentos,  axes[1], "Matching e Christofides (lentos)"),
        ]:
            for alg, d in sorted(grupo.items()):
                ns     = sorted(d.keys())
                tempos = [d[n] for n in ns]
                ax.plot(ns, tempos, 'o-', label=alg, color=cores.get(alg, 'gray'),
                        linewidth=1.5, markersize=4)
            ax.set_xlabel("n (vertices)")
            ax.set_ylabel("Tempo medio (ms)")
            ax.set_title(titulo)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        arq = os.path.join(DIR_GRAFICOS, "tempo_separado.png")
        plt.savefig(arq, dpi=120)
        plt.close()
        print(f"  Salvo: {arq}")


# ───────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ───────────────────────────────────────────────────────────

def executar_stress_test(config):
    """
    Executa o stress test conforme a configuração e grava resultados
    de forma incremental.

    config deve ter os atributos:
        tamanhos, distribuicoes, algoritmos, timeout, repetitions,
        max_memory, stop_on_timeout, seed
    """
    global _interrompido

    os.makedirs(DIR_RESULTADOS, exist_ok=True)
    os.makedirs(DIR_GRAFICOS, exist_ok=True)

    salvador = SalvadorCSV(ARQ_CSV)
    resultados_por_n = {}

    esgotado = {alg: False for alg in config.algoritmos}

    print(f"\n{'='*58}")
    print("STRESS TEST — TP2 Algoritmos 2")
    print(f"{'='*58}")
    print(f"  Algoritmos   : {config.algoritmos}")
    print(f"  Tamanhos     : n={config.tamanhos[0]} ate n={config.tamanhos[-1]} "
          f"({len(config.tamanhos)} pontos)")
    print(f"  Distribuicoes: {config.distribuicoes}")
    print(f"  Timeout      : {config.timeout}s por execucao")
    print(f"  Repeticoes   : {config.repetitions}")
    print(f"  Parar em TO  : {'sim' if config.stop_on_timeout else 'nao'}")
    print(f"  psutil (RSS) : {'disponivel' if PSUTIL_DISPONIVEL else 'nao disponivel'}")
    print(f"  Resultados   : {ARQ_CSV}")
    print(f"{'='*58}")

    # Inicia processo filho (spawn + importar modulos: ~200-500ms no Windows)
    print("  Iniciando processo filho...")
    gerenciador = GerenciadorProcesso()
    print(f"  Processo filho pronto (PID: {gerenciador._processo.pid})")

    t_inicio_total = time.perf_counter()

    try:
        for n in config.tamanhos:
            if _interrompido:
                break

            ativos = [alg for alg in config.algoritmos
                      if not esgotado[alg] or not config.stop_on_timeout]
            if config.stop_on_timeout and not ativos:
                print(f"\n  Todos os algoritmos esgotados antes de n={n}. Encerrando.")
                break

            print(f"\n{'─'*58}")
            print(f"  N = {n}")
            print(f"{'─'*58}")
            resultados_por_n[n] = []

            for dist_tipo in config.distribuicoes:
                if _interrompido:
                    break

                for rep in range(1, config.repetitions + 1):
                    if _interrompido:
                        break

                    # Sementes diferentes por repetição: evita correlação entre reps
                    seed_inst = config.seed + rep * 10000 + n
                    coords    = gerar_instancia(dist_tipo, n, seed=seed_inst)

                    t0  = time.perf_counter()
                    mat = montar_matriz(coords, n)
                    t_mat_ms = (time.perf_counter() - t0) * 1000

                    for alg in config.algoritmos:
                        if _interrompido:
                            break

                        if esgotado[alg] and config.stop_on_timeout:
                            linha = {
                                'n': n, 'distribuicao': dist_tipo, 'semente': seed_inst,
                                'algoritmo': alg, 'repeticao': rep,
                                't_total_ms': 0, 't_mst_ms': 0,
                                't_matriz_ms': round(t_mat_ms, 4),
                                'mem_rss_pico_mb': -1,
                                'n_vertices_impares': -1, 'peso_mst': 0,
                                'custo_rota': 0, 'rota_valida': False, 'status': 'ESGOTADO',
                            }
                            salvador.gravar(linha)
                            continue

                        metricas = executar_algoritmo(
                            alg, mat, n, gerenciador, config.timeout, config.max_memory
                        )

                        linha = {
                            'n': n, 'distribuicao': dist_tipo, 'semente': seed_inst,
                            'algoritmo': alg, 'repeticao': rep,
                            't_total_ms':         round(metricas['t_total_ms'], 4),
                            't_mst_ms':           round(metricas['t_mst_ms'], 4),
                            't_matriz_ms':        round(t_mat_ms, 4),
                            'mem_rss_pico_mb':    round(metricas['mem_rss_pico_mb'], 3),
                            'n_vertices_impares': metricas['n_vertices_impares'],
                            'peso_mst':           metricas['peso_mst'],
                            'custo_rota':         metricas['custo_rota'],
                            'rota_valida':        metricas['rota_valida'],
                            'status':             metricas['status'],
                        }
                        salvador.gravar(linha)
                        resultados_por_n[n].append(linha)

                        status = metricas['status']
                        t_ms   = metricas['t_total_ms']
                        emoji  = "OK" if status == 'OK' else "!!"

                        if rep == 1:
                            print(f"  [{emoji}] {alg:<15} dist={dist_tipo:<13} "
                                  f"n={n:>5}  {t_ms:>10.1f}ms  [{status}]")

                        if status in ('TIMEOUT', 'MEMORIA_EXCEDIDA') or status.startswith('ERRO'):
                            esgotado[alg] = True
                            if config.stop_on_timeout:
                                print(f"    -> {alg} esgotado em n={n} ({status}). "
                                      f"Sera pulado para tamanhos maiores.")

            atualizar_resumo(resultados_por_n, config.algoritmos)

    except KeyboardInterrupt:
        _interrompido = True
        print("\n\n  [CTRL+C] Interrupcao solicitada -- salvando resultados...")

    finally:
        gerenciador.fechar()
        salvador.fechar()

    duracao = time.perf_counter() - t_inicio_total
    print(f"\n{'='*58}")
    if _interrompido:
        print("  Teste interrompido (resultados parciais salvos).")
    else:
        print("  Stress test concluido.")
    print(f"  Duracao total : {duracao:.1f}s")
    print(f"{'='*58}")
    print("  Ultimos N testados por algoritmo:")
    for alg in config.algoritmos:
        ns_ok = sorted(set(
            r['n'] for linhas in resultados_por_n.values()
            for r in linhas if r['algoritmo'] == alg and r['status'] == 'OK'
        ))
        ultimo = ns_ok[-1] if ns_ok else 0
        print(f"    {alg:<15}: n={ultimo}")

    print(f"\n  Gerando graficos...")
    gerar_graficos(resultados_por_n, config.algoritmos)

    print(f"\n  Arquivos salvos em: {DIR_RESULTADOS}/")
    print(f"    {ARQ_CSV}")
    print(f"    {ARQ_RESUMO}")
    print(f"    {DIR_GRAFICOS}/tempo_todos.png")
    print(f"    {DIR_GRAFICOS}/tempo_separado.png")


# ───────────────────────────────────────────────────────────
# VALIDAÇÃO DE INICIALIZAÇÃO
# ───────────────────────────────────────────────────────────

def validar_inicializacao():
    """
    Verificação rápida: importa todos os módulos e executa uma instância
    de 4 vértices em cada algoritmo. Não realiza testes de desempenho.
    Roda direto no processo principal (sem subprocess) para ser instantânea.
    """
    import math

    print("\nValidacao de inicializacao -- TP2 Stress Test")
    print("=" * 50)

    coords = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
    n = 4
    mat = montar_matriz(coords, n)

    resultados = []

    try:
        arestas = prim(mat, n)
        peso    = peso_total_mst(arestas)
        resultados.append(('MST (Prim)', 'OK', f'peso={peso:.2f}'))
    except Exception as e:
        resultados.append(('MST (Prim)', 'ERRO', str(e)))

    try:
        rota, custo, _ = twice_around_the_tree(mat, n)
        ok = validar_rota(rota, n, mat)
        resultados.append(('Twice-around', 'OK', f'custo={custo:.2f} rota_valida={ok}'))
    except Exception as e:
        resultados.append(('Twice-around', 'ERRO', str(e)))

    try:
        arestas_mst = prim(mat, n)
        am, pm = emparelhamento_perfeito_minimo(mat, n, arestas_mst)
        resultados.append(('Matching', 'OK', f'peso_match={pm:.2f}'))
    except Exception as e:
        resultados.append(('Matching', 'ERRO', str(e)))

    try:
        rota, custo, _, pm = christofides(mat, n)
        ok = validar_rota(rota, n, mat)
        resultados.append(('Christofides', 'OK', f'custo={custo:.2f} rota_valida={ok}'))
    except Exception as e:
        resultados.append(('Christofides', 'ERRO', str(e)))

    passou = sum(1 for _, s, _ in resultados if s == 'OK')
    falhou = len(resultados) - passou

    for nome, status, detalhe in resultados:
        simbolo = "+" if status == 'OK' else "X"
        print(f"  [{simbolo}] {nome:<15}  {status}  ({detalhe})")

    print(f"\n  psutil disponivel : {'sim (memoria RSS ativa)' if PSUTIL_DISPONIVEL else 'nao (instale: pip install psutil)'}")
    print(f"\n  Resultado: {passou}/{len(resultados)} algoritmos funcionando corretamente.")

    if falhou > 0:
        print("\n  ATENCAO: Algum algoritmo falhou. Verifique se esta rodando da raiz do projeto.")
        print("  Exemplo: cd <pasta_do_projeto> && python -m tests.stress_test --validate")
        return False

    print("\n  Tudo pronto! Use um dos comandos abaixo para iniciar os testes:")
    print("    python -m tests.stress_test --mode quick")
    print("    python -m tests.stress_test --algorithm all --min-n 10 --max-n 1000 --step 50 --timeout 120 --repetitions 3")
    return True


# ───────────────────────────────────────────────────────────
# ARGPARSE E MAIN
# ───────────────────────────────────────────────────────────

class Config:
    """Agrupa as opções de configuração do stress test."""
    pass


def main():
    parser = argparse.ArgumentParser(
        prog='python -m tests.stress_test',
        description="Stress Test dos algoritmos do TP2 -- TSP (DCC207 / UFMG)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemplos:
  python -m tests.stress_test --validate
  python -m tests.stress_test --mode quick
  python -m tests.stress_test --algorithm all --min-n 10 --max-n 2000 --step 25 --timeout 120 --repetitions 3
  python -m tests.stress_test --algorithm christofides --min-n 10 --max-n 500 --timeout 120
  python -m tests.stress_test --algorithm matching --min-n 10 --max-n 300 --timeout 120
  python -m tests.stress_test --distribution uniform circle --max-n 500 --repetitions 5
        """
    )

    parser.add_argument('--validate', action='store_true',
                        help='Verifica imports e executa instancia n=4 sem rodar testes completos.')
    parser.add_argument('--mode', choices=['quick', 'full'], default=None,
                        help='Modo predefinido:\n'
                             '  quick: n=10..100, passo=25, timeout=10s, 1 rep, so uniform\n'
                             '  full : n=10..1000, passo=50, timeout=120s, 3 reps, todas distribuicoes')

    parser.add_argument('--algorithm', nargs='+', default=['all'],
                        metavar='ALG',
                        help='Algoritmos: all | mst | twice | matching | christofides\n'
                             '  Pode especificar varios: --algorithm mst twice')
    parser.add_argument('--min-n', type=int, default=10,
                        help='Tamanho inicial de N (padrao: 10)')
    parser.add_argument('--max-n', type=int, default=300,
                        help='Tamanho maximo de N (padrao: 300)')
    parser.add_argument('--step', type=int, default=25,
                        help='Incremento de N (padrao: 25)')
    parser.add_argument('--distribution', nargs='+', default=['uniform'],
                        metavar='DIST',
                        help='Distribuicoes: all | uniform | circle | line | elongated |\n'
                             '               clusters | convex_few | convex_many | irregular\n'
                             '  Pode especificar varias: --distribution uniform circle clusters')

    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Tempo maximo por execucao em segundos (padrao: 60)')
    parser.add_argument('--repetitions', type=int, default=3,
                        help='Repeticoes por ponto (padrao: 3)')
    parser.add_argument('--max-memory', type=float, default=None,
                        metavar='MB',
                        help='Limite de memoria por execucao em MB (requer psutil).\n'
                             '  Monitora RSS do processo filho a cada 100ms e encerra\n'
                             '  o processo (kill real) se exceder o limite.')
    parser.add_argument('--continue-on-timeout', action='store_true', default=False,
                        help='Continua testando tamanhos maiores mesmo apos timeout.\n'
                             '  Por padrao, um algoritmo que atingiu timeout e pulado\n'
                             '  para todos os N seguintes.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semente aleatoria base (padrao: 42)')

    args = parser.parse_args()

    if args.validate:
        ok = validar_inicializacao()
        sys.exit(0 if ok else 1)

    if args.mode == 'quick':
        args.min_n        = 10
        args.max_n        = 100
        args.step         = 25
        args.timeout      = 10.0
        args.repetitions  = 1
        args.distribution = ['uniform']
        print("Modo rapido: n=10..100, timeout=10s, 1 repeticao, distribuicao=uniform")

    elif args.mode == 'full':
        args.min_n        = 10
        args.max_n        = 1000
        args.step         = 50
        args.timeout      = 120.0
        args.repetitions  = 3
        args.distribution = ['all']
        print("Modo completo: n=10..1000, timeout=120s, 3 repeticoes, todas distribuicoes")

    if 'all' in args.algorithm:
        algoritmos = TODOS_ALGORITMOS[:]
    else:
        algoritmos = [a for a in args.algorithm if a in TODOS_ALGORITMOS]
        invalidos  = [a for a in args.algorithm if a not in TODOS_ALGORITMOS and a != 'all']
        if invalidos:
            print(f"  [aviso] Algoritmos desconhecidos ignorados: {invalidos}")

    if 'all' in args.distribution:
        distribuicoes = TODAS_DISTRIBUICOES[:]
    else:
        distribuicoes = [d for d in args.distribution if d in TODAS_DISTRIBUICOES]
        invalidos     = [d for d in args.distribution if d not in TODAS_DISTRIBUICOES and d != 'all']
        if invalidos:
            print(f"  [aviso] Distribuicoes desconhecidas ignoradas: {invalidos}")

    if not algoritmos:
        print("ERRO: Nenhum algoritmo valido selecionado.")
        sys.exit(1)
    if not distribuicoes:
        print("ERRO: Nenhuma distribuicao valida selecionada.")
        sys.exit(1)

    min_n    = args.min_n
    max_n    = args.max_n
    step     = args.step
    tamanhos = list(range(min_n, max_n + 1, step))
    if not tamanhos or tamanhos[-1] != max_n:
        tamanhos.append(max_n)
    tamanhos = sorted(set(tamanhos))

    config = Config()
    config.tamanhos         = tamanhos
    config.distribuicoes    = distribuicoes
    config.algoritmos       = algoritmos
    config.timeout          = args.timeout
    config.repetitions      = args.repetitions
    config.max_memory       = args.max_memory
    config.stop_on_timeout  = not args.continue_on_timeout
    config.seed             = args.seed

    def handler_sinal(sig, frame):
        global _interrompido
        _interrompido = True

    try:
        signal.signal(signal.SIGINT, handler_sinal)
    except (OSError, ValueError):
        pass

    executar_stress_test(config)


if __name__ == '__main__':
    main()

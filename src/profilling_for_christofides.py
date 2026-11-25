import cProfile
import csv
import io
import os
import pstats
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from christofides import christofides
from instancias import INSTANCIAS, _achar_arquivo
from medicao import medir_tempo, medir_memoria_kb
from twice_around import twice_around_the_tree
from utils import ler_instancia_tsplib_completa, calcular_matriz_distancias

ETAPAS = ['mst', 'impares', 'matching', 'multigrafo', 'euler', 'shortcut']
BASES_SWEEP = ['st70', 'berlin52']

COLUNAS = (
    ['instancia', 'n', 'tipo', 'repeticoes']
    + [f't_{e}_ms' for e in ETAPAS]
    + ['t_christofides_ms', 't_christofides_nogc_ms', 'pct_matching',
       'mem_christofides_kb', 't_twice_ms', 'mem_twice_kb']
)


def montar_conjunto(pasta, passo):
    """Lista de (nome, n, tipo, distancias): instâncias do TP + varredura em N."""
    conjunto, vistos, cache = [], set(), {}

    def base(nome):
        if nome not in cache:
            try:
                cache[nome] = ler_instancia_tsplib_completa(_achar_arquivo(pasta, nome))
            except FileNotFoundError as e:
                print(f"AVISO: {e}; ignorada")
                cache[nome] = None
        return cache[nome]

    def adicionar(nome, n, tipo, dist):
        if nome not in vistos:
            vistos.add(nome)
            conjunto.append((nome, n, tipo, dist))

    for e in INSTANCIAS:
        d = base(e.get('base', e['nome']))
        if d is None:
            continue
        if 'k' in e:
            k = e['k']
            coords = {i: d['coords'][i] for i in range(k)}
            adicionar(e['nome'], k, d['tipo'], calcular_matriz_distancias(coords, k, d['tipo']))
        else:
            adicionar(e['nome'], d['n'], d['tipo'], d['distancias'])

    for nome_base in BASES_SWEEP:  # varredura em N (só bases com coordenadas reais)
        d = base(nome_base)
        if d is None or not d['coords_reais']:
            continue
        tamanhos = list(range(10, d['n'], passo))
        for k in tamanhos:
            coords = {i: d['coords'][i] for i in range(k)}
            adicionar(f"{nome_base}_N{k}", k, d['tipo'],
                      calcular_matriz_distancias(coords, k, d['tipo']))
    return conjunto


def medir_instancia(nome, n, tipo, dist, reps):
    perfis, totais = [], []
    for _ in range(reps):
        p = {}
        t0 = time.perf_counter()
        christofides(dist, n, perfil=p)
        totais.append(time.perf_counter() - t0)
        perfis.append(p)
    med = {e: statistics.median(p[e] for p in perfis) for e in ETAPAS}
    t_chr = statistics.median(totais)

    _, t_nogc = medir_tempo(christofides, dist, n, repeticoes=reps, desabilitar_gc=True)
    _, mem_chr = medir_memoria_kb(christofides, dist, n)
    _, t_twice = medir_tempo(twice_around_the_tree, dist, n, repeticoes=reps)
    _, mem_twice = medir_memoria_kb(twice_around_the_tree, dist, n)

    linha = {'instancia': nome, 'n': n, 'tipo': tipo, 'repeticoes': reps}
    for e in ETAPAS:
        linha[f't_{e}_ms'] = round(med[e] * 1000, 5)
    linha.update({
        't_christofides_ms': round(t_chr * 1000, 5),
        't_christofides_nogc_ms': round(t_nogc * 1000, 5),
        'pct_matching': round(100 * med['matching'] / sum(med.values()), 2),
        'mem_christofides_kb': round(mem_chr, 2),
        't_twice_ms': round(t_twice * 1000, 5),
        'mem_twice_kb': round(mem_twice, 2),
    })
    return linha


def cprofile_maior(conjunto, arquivo):
    nome, n, _, dist = max(conjunto, key=lambda c: c[1])
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        christofides(dist, n)
    pr.disable()
    buf = io.StringIO()
    buf.write(f"cProfile do Christofides em {nome} (n={n}), 20 execuções\n\n")
    pstats.Stats(pr, stream=buf).sort_stats('cumulative').print_stats(25)
    with open(arquivo, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    print(f"cProfile salvo em: {arquivo}")


def main(pasta='data', reps=7, passo=5, saida='results/profiling_subrotinas.csv'):
    os.makedirs(os.path.dirname(saida) or '.', exist_ok=True)
    conjunto = montar_conjunto(pasta, passo)
    if not conjunto:
        print("Nenhuma instância encontrada.")
        return
    with open(saida, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        for nome, n, tipo, dist in sorted(conjunto, key=lambda c: c[1]):
            linha = medir_instancia(nome, n, tipo, dist, reps)
            w.writerow(linha)
            f.flush()
            print(f"  {nome:<16} n={n:<3} total={linha['t_christofides_ms']:>9.3f} ms "
                  f"matching={linha['pct_matching']:>5.1f}% "
                  f"mem={linha['mem_christofides_kb']:.0f} KB")
    print(f"\nPerfil salvo em: {saida}")
    cprofile_maior(conjunto, os.path.join(os.path.dirname(saida) or '.', 'cprofile_christofides.txt'))


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else 'data',
        int(sys.argv[2]) if len(sys.argv) > 2 else 7,
        int(sys.argv[3]) if len(sys.argv) > 3 else 5,
    )
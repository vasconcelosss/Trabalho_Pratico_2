import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from christofides import christofides
from features import _carregar_com_meta
from heuristics import nearest_neighbor, convex_hull_insertion
from instancias import INSTANCIAS
from twice_around import twice_around_the_tree

COLUNAS = [
    'instancia', 'n', 'faixa',
    'otimo_bnb', 'otimo_tabela', 'otimo_usado', 'fonte_otimo', 'conferencia',
    'custo_twice', 'custo_christofides', 'custo_nn', 'custo_hull',
    'gap_twice_pct', 'gap_christofides_pct', 'gap_nn_pct', 'gap_hull_pct',
    'triangulo_ok', 'viola_garantia', 'alerta',
]


def carregar_otimos_bnb(caminho):
    """nome -> (ótimo ou None, status) a partir do CSV do B&B."""
    otimos = {}
    if not os.path.exists(caminho):
        print(f"AVISO: {caminho} não existe; só a tabela será usada.")
        return otimos
    with open(caminho, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            nome = r['instancia']
            if r['status'] == 'OK' and r['custo_otimo'] not in ('NA', ''):
                otimos.setdefault(nome, set()).add(float(r['custo_otimo']))
            else:
                otimos.setdefault(nome, set())
    resultado = {}
    for nome, valores in otimos.items():
        if not valores:
            resultado[nome] = (None, 'TIMEOUT')
        elif len(valores) == 1:
            resultado[nome] = (next(iter(valores)), 'OK')
        else:
            resultado[nome] = (min(valores), 'INCONSISTENTE_BNB')
    return resultado


def respeita_triangulo(d, n):
    for i in range(n):
        for j in range(n):
            dij = d[i][j]
            for k in range(n):
                if dij > d[i][k] + d[k][j]:
                    return False
    return True


def _gap(custo, otimo):
    if custo is None or otimo is None or otimo <= 0:
        return None
    return (custo - otimo) / otimo * 100.0


def _fmt(v, casas=4):
    return 'NA' if v is None else (round(v, casas) if isinstance(v, float) else v)


def calcular_gaps(pasta='data', bnb_csv='results/resultados_bnb.csv', saida='results/gaps.csv'):
    otimos_bnb = carregar_otimos_bnb(bnb_csv)
    tabela = {e['nome']: e['otimo'] for e in INSTANCIAS}
    os.makedirs(os.path.dirname(saida) or '.', exist_ok=True)

    linhas = []
    for inst in _carregar_com_meta(pasta):
        nome, n, dist, coords = inst['nome'], inst['n'], inst['distancias'], inst['coords']

        _, c_twice = twice_around_the_tree(dist, n)[:2]
        _, c_chr = christofides(dist, n)[:2]
        _, c_nn = nearest_neighbor(dist, n)
        c_hull = None
        if coords and len(coords) >= n:
            _, c_hull = convex_hull_insertion(dist, n, coords)

        o_bnb, st_bnb = otimos_bnb.get(nome, (None, 'SEM_EXPERIMENTO'))
        o_tab = tabela.get(nome)

        if st_bnb == 'INCONSISTENTE_BNB':
            conferencia = 'INCONSISTENTE_BNB'
        elif o_bnb is not None and o_tab is not None:
            conferencia = 'OK' if abs(o_bnb - o_tab) < 1e-6 else 'DIVERGE'
        elif o_bnb is not None:
            conferencia = 'SEM_TABELA'
        elif st_bnb == 'TIMEOUT':
            conferencia = 'TIMEOUT_BNB'
        else:
            conferencia = 'SEM_BNB'

        if o_bnb is not None:
            o_usado, fonte = o_bnb, 'bnb'
        elif o_tab is not None:
            o_usado, fonte = float(o_tab), 'tabela'
        else:
            o_usado, fonte = None, 'NA'

        tri = respeita_triangulo(dist, n)
        g_twice, g_chr = _gap(c_twice, o_usado), _gap(c_chr, o_usado)
        viola = None
        if tri and o_usado is not None:
            viola = (c_twice > 2 * o_usado + 1e-9) or (c_chr > 1.5 * o_usado + 1e-9)

        # qualquer rota válida custa >= ótimo; se não, o ótimo usado está errado
        # (tipo de distância errado, tabela incompatível ou bug no B&B)
        custos = [c for c in (c_twice, c_chr, c_nn, c_hull) if c is not None]
        alerta = ('GAP_NEGATIVO' if o_usado is not None and min(custos) < o_usado - 1e-9
                  else 'NA')

        linhas.append({
            'instancia': nome, 'n': n, 'faixa': inst['faixa'],
            'otimo_bnb': _fmt(o_bnb), 'otimo_tabela': _fmt(o_tab), 'otimo_usado': _fmt(o_usado),
            'fonte_otimo': fonte, 'conferencia': conferencia,
            'custo_twice': c_twice, 'custo_christofides': c_chr,
            'custo_nn': c_nn, 'custo_hull': _fmt(c_hull),
            'gap_twice_pct': _fmt(g_twice), 'gap_christofides_pct': _fmt(g_chr),
            'gap_nn_pct': _fmt(_gap(c_nn, o_usado)), 'gap_hull_pct': _fmt(_gap(c_hull, o_usado)),
            'triangulo_ok': tri, 'viola_garantia': _fmt(viola), 'alerta': alerta,
        })

    with open(saida, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS)
        w.writeheader()
        w.writerows(linhas)

    def _p(v):
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    print(f"{'instância':<14}{'n':>4} {'ótimo':>8} {'fonte':<7} {'conferência':<13}"
          f"{'twice':>9}{'christ.':>9}{'nn':>9}{'hull':>9}  alerta")
    for r in sorted(linhas, key=lambda x: x['n']):
        print(f"{r['instancia']:<14}{r['n']:>4} {_p(r['otimo_usado']):>8} {r['fonte_otimo']:<7} "
              f"{r['conferencia']:<13}{_p(r['gap_twice_pct']):>9}{_p(r['gap_christofides_pct']):>9}"
              f"{_p(r['gap_nn_pct']):>9}{_p(r['gap_hull_pct']):>9}  {r['alerta']}")
    print("(gaps em %; NA = ótimo desconhecido)")

    for col, rotulo in (('gap_twice_pct', 'Twice-around'), ('gap_christofides_pct', 'Christofides')):
        v = [r[col] for r in linhas if r[col] != 'NA']
        if v:
            print(f"\n{rotulo}: gap médio {sum(v)/len(v):.2f}% | máx {max(v):.2f}% | n={len(v)}")

    problemas = [r['instancia'] for r in linhas
                 if r['conferencia'] in ('DIVERGE', 'INCONSISTENTE_BNB') or r['alerta'] != 'NA']
    if problemas:
        print(f"\n!! Conferir (B&B diverge da tabela/entre estratégias ou gap negativo): {problemas}")
    print(f"\nGaps salvos em: {saida}")


if __name__ == '__main__':
    calcular_gaps(
        sys.argv[1] if len(sys.argv) > 1 else 'data',
        sys.argv[2] if len(sys.argv) > 2 else 'results/resultados_bnb.csv',
        sys.argv[3] if len(sys.argv) > 3 else 'results/gaps.csv',
    )
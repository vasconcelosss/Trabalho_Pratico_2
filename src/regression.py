import csv
import math
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

MODELOS = {
    'a*N + c':       lambda n: n,
    'a*N*logN + c':  lambda n: n * np.log(n),
    'a*N^2 + c':     lambda n: n ** 2,
    'a*N^3 + c':     lambda n: n ** 3,
}

ALVOS = [
    ('t_christofides_ms', 'Christofides: tempo total (ms)'),
    ('t_christofides_nogc_ms', 'Christofides: tempo total sem GC (ms)'),
    ('t_matching_ms', 'Christofides: matching (ms)'),
    ('t_mst_ms', 'Christofides: MST/Prim (ms)'),
    ('t_euler_ms', 'Christofides: circuito euleriano (ms)'),
    ('t_twice_ms', 'Twice-around: tempo total (ms)'),
    ('mem_christofides_kb', 'Christofides: pico de memória (KB)'),
    ('mem_twice_kb', 'Twice-around: pico de memória (KB)'),
]


def _metricas(y, yp):
    y, yp = np.asarray(y, float), np.asarray(yp, float)
    ss_res = float(((y - yp) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = math.sqrt(ss_res / len(y))
    return r2, rmse


def ajustar_modelos(n, y):
    """Lista de dicts com modelo, parâmetros, R2 e RMSE."""
    n, y = np.asarray(n, float), np.asarray(y, float)
    saida = []
    for nome, f in MODELOS.items():
        try:
            x = f(n)
            esc = max(x.max(), 1.0)
            (a, c), _ = curve_fit(lambda xx, a, c: a * xx + c, x / esc, y)
            a = a / esc
            r2, rmse = _metricas(y, a * x + c)
            saida.append({'modelo': nome, 'a': a, 'c_ou_b': c, 'r2': r2, 'rmse': rmse})
        except Exception as e:  # noqa: BLE001
            print(f"  aviso: ajuste '{nome}' falhou: {e}")

    try:
        mask = y > 0
        b0, loga0 = np.polyfit(np.log(n[mask]), np.log(y[mask]), 1)
        (a, b), _ = curve_fit(lambda nn, a, b: a * nn ** b, n, y,
                              p0=[math.exp(loga0), b0], maxfev=20000)
        r2, rmse = _metricas(y, a * n ** b)
        saida.append({'modelo': 'a*N^b', 'a': a, 'c_ou_b': b, 'r2': r2, 'rmse': rmse})
    except Exception as e:  # noqa: BLE001
        print(f"  aviso: lei de potência falhou: {e}")
    return saida


def melhor(ajustes):
    validos = [a for a in ajustes if a['a'] > 0 and not math.isnan(a['r2'])]
    return max(validos, key=lambda a: a['r2']) if validos else None


def _ler_csv(caminho):
    if not os.path.exists(caminho):
        return []
    with open(caminho, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _plot(n, y, ajustes, titulo, arquivo):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.dirname(arquivo), exist_ok=True)
    nn = np.linspace(min(n), max(n), 200)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(n, y, color='black', zorder=3, label='medido')
    for a in ajustes:
        if a['modelo'] == 'a*N^b':
            yy = a['a'] * nn ** a['c_ou_b']
            rot = f"N^{a['c_ou_b']:.2f} (R²={a['r2']:.3f})"
        else:
            f = MODELOS[a['modelo']]
            yy = a['a'] * f(nn) + a['c_ou_b']
            rot = f"{a['modelo'].split('+')[0].strip()} (R²={a['r2']:.3f})"
        ax.plot(nn, yy, label=rot, linewidth=1.2)
    ax.set_xlabel('N (vértices)')
    ax.set_title(titulo, fontsize=10)
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)


def _plot_resumos_aproximados(rows, ajustes_por_col, pasta_fig):
    """Figuras consolidadas prontas para o relatório SBC."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(pasta_fig, exist_ok=True)
    rows_ord = sorted(rows, key=lambda r: float(r['n']))
    n = np.array([float(r['n']) for r in rows_ord])
    nn = np.linspace(n.min(), n.max(), 200)

    # 1) Christofides: com vs sem GC + participação do Matching
    t_gc = np.array([_num(r.get('t_christofides_ms')) or np.nan for r in rows_ord])
    t_nogc = np.array([_num(r.get('t_christofides_nogc_ms')) or np.nan for r in rows_ord])
    pct_match = np.array([_num(r.get('pct_matching')) or np.nan for r in rows_ord])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.scatter(n, t_gc, color='#c0392b', alpha=0.8, s=25, label='Com GC (medido)')
    ax1.scatter(n, t_nogc, color='#2980b9', alpha=0.8, s=25, marker='^', label='Sem GC (medido)')
    for col, cor, rot in (('t_christofides_ms', '#c0392b', 'Com GC'),
                          ('t_christofides_nogc_ms', '#2980b9', 'Sem GC')):
        m = melhor(ajustes_por_col.get(col, []))
        if m:
            if m['modelo'] == 'a*N^b':
                yy = m['a'] * nn ** m['c_ou_b']
                lbl = f"{rot}: N^{m['c_ou_b']:.2f} (R²={m['r2']:.3f})"
            else:
                yy = m['a'] * MODELOS[m['modelo']](nn) + m['c_ou_b']
                lbl = f"{rot}: {m['modelo'].split('+')[0].strip()} (R²={m['r2']:.3f})"
            ax1.plot(nn, yy, color=cor, linewidth=1.6, label=lbl)
    ax1.set_xlabel('N (vértices)')
    ax1.set_ylabel('Tempo (ms)')
    ax1.set_title('Christofides: Impacto do Garbage Collector')
    ax1.grid(alpha=0.25, linestyle='--')
    ax1.legend(fontsize=7.5)

    ax2.plot(n, pct_match, 'o-', color='#8e44ad', markersize=4, linewidth=1.2)
    ax2.set_xlabel('N (vértices)')
    ax2.set_ylabel('% do tempo gasto no Matching')
    ax2.set_ylim(min(80, float(np.nanmin(pct_match)) - 2), 100.5)
    ax2.set_title('Gargalo: Participação do Matching no Christofides')
    ax2.grid(alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, 'resumo_christofides_gc_e_subrotinas.png'), dpi=160)
    plt.close(fig)

    # 2) Memória: Christofides vs Twice-around
    mem_c = np.array([_num(r.get('mem_christofides_kb')) or np.nan for r in rows_ord])
    mem_t = np.array([_num(r.get('mem_twice_kb')) or np.nan for r in rows_ord])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(n, mem_c, color='#d35400', s=25, label='Christofides (medido)')
    ax.scatter(n, mem_t, color='#27ae60', s=25, marker='s', label='Twice-around (medido)')
    for col, cor, rot in (('mem_christofides_kb', '#d35400', 'Christofides'),
                          ('mem_twice_kb', '#27ae60', 'Twice-around')):
        m = melhor(ajustes_por_col.get(col, []))
        if m and m['modelo'] == 'a*N^b':
            ax.plot(nn, m['a'] * nn ** m['c_ou_b'], color=cor, linewidth=1.5,
                    label=f"{rot}: N^{m['c_ou_b']:.2f} (R²={m['r2']:.3f})")
    ax.set_xlabel('N (vértices)')
    ax.set_ylabel('Pico de Memória (KB)')
    ax.set_title('Complexidade Espacial Empírica: Christofides vs Twice-around')
    ax.grid(alpha=0.25, linestyle='--')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, 'resumo_memoria.png'), dpi=160)
    plt.close(fig)


def _plot_bnb(rows, pasta_fig):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(pasta_fig, exist_ok=True)
    estrategias = sorted({r['estrategia'] for r in rows})
    if not estrategias:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    cores = {
        'cold_start': '#7f8c8d',
        'nearest_neighbor': '#2980b9',
        'convex_hull': '#27ae60',
        'twice_around': '#f39c12',
        'christofides': '#c0392b',
    }
    marcadores = {
        'cold_start': 'o',
        'nearest_neighbor': 's',
        'convex_hull': '^',
        'twice_around': 'D',
        'christofides': '*',
    }

    for est in estrategias:
        sub = sorted([r for r in rows if r['estrategia'] == est], key=lambda r: float(r['n']))
        n = np.array([float(r['n']) for r in sub])
        exp = np.array([max(1.0, float(r['nos_expandidos'])) for r in sub])
        t_ms = np.array([max(1e-3, float(r['t_bnb_ms'])) for r in sub])
        cor = cores.get(est, None)
        mk = marcadores.get(est, 'o')
        ax1.plot(n, exp, marker=mk, color=cor, linewidth=1.3, markersize=5, label=est)
        ax2.plot(n, t_ms, marker=mk, color=cor, linewidth=1.3, markersize=5, label=est)

    for ax, ylab, tit in ((ax1, 'Nós expandidos (escala log)', 'B&B: Nós Expandidos por Warm-Start'),
                          (ax2, 'Tempo B&B em ms (escala log)', 'B&B: Tempo de Execução por Warm-Start')):
        ax.set_yscale('log')
        ax.set_xlabel('N (vértices)')
        ax.set_ylabel(ylab)
        ax.set_title(tit)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.legend(fontsize=7.5)

    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, 'bnb_escalabilidade.png'), dpi=160)
    plt.close(fig)


def parte_aproximados(perfil_csv, linhas_saida, pasta_fig):
    rows = _ler_csv(perfil_csv)
    if not rows:
        print(f"Sem dados em {perfil_csv} (rode profilling_for_christofides.py antes).")
        return
    print(f"\n=== Aproximados ({len(rows)} pontos, N de "
          f"{min(int(r['n']) for r in rows)} a {max(int(r['n']) for r in rows)}) ===")
    ajustes_por_col = {}
    for col, titulo in ALVOS:
        pts = [(float(r['n']), _num(r[col])) for r in rows if _num(r.get(col)) is not None]
        if len(pts) < 5:
            continue
        n, y = zip(*pts)
        ajustes = ajustar_modelos(n, y)
        ajustes_por_col[col] = ajustes
        m = melhor(ajustes)
        print(f"\n{titulo}")
        for a in ajustes:
            marca = ' <- melhor' if m is a else ''
            extra = f"b={a['c_ou_b']:.3f}" if a['modelo'] == 'a*N^b' else f"c={a['c_ou_b']:.4g}"
            print(f"  {a['modelo']:<14} a={a['a']:.3e} {extra:<12} "
                  f"R2={a['r2']:.4f} RMSE={a['rmse']:.4g}{marca}")
            linhas_saida.append({'parte': 'aproximados', 'grupo': col, 'modelo': a['modelo'],
                                 'a': a['a'], 'c_ou_b': a['c_ou_b'], 'r2': a['r2'],
                                 'rmse': a['rmse'], 'n_pontos': len(n), 'melhor': m is a})
        _plot(np.array(n), np.array(y), ajustes, titulo, os.path.join(pasta_fig, f"{col}.png"))

    _plot_resumos_aproximados(rows, ajustes_por_col, pasta_fig)


def parte_bnb(bnb_csv, linhas_saida, pasta_fig):
    rows = [r for r in _ler_csv(bnb_csv) if r['status'] == 'OK']
    if not rows:
        print(f"\nSem execuções OK do B&B em {bnb_csv}.")
        return
    print("\n=== B&B: crescimento exponencial y = a*exp(b*N), por estratégia ===")
    for estrategia in sorted({r['estrategia'] for r in rows}):
        for col in ('nos_expandidos', 't_bnb_ms'):
            pts = [(float(r['n']), _num(r[col])) for r in rows
                   if r['estrategia'] == estrategia and (_num(r[col]) or 0) > 0]
            if len(pts) < 4:
                continue
            n, y = map(np.array, zip(*pts))
            b, loga = np.polyfit(n, np.log(y), 1)
            r2, _ = _metricas(np.log(y), loga + b * n)
            _, rmse = _metricas(y, math.exp(loga) * np.exp(b * n))
            print(f"  {estrategia:<17} {col:<15} b={b:.3f} (x{math.exp(b):.2f} por vértice) "
                  f"R2(log)={r2:.3f} RMSE={rmse:.4g} pontos={len(n)}")
            linhas_saida.append({'parte': 'bnb', 'grupo': f"{estrategia}:{col}",
                                 'modelo': 'a*exp(b*N)', 'a': math.exp(loga), 'c_ou_b': b,
                                 'r2': r2, 'rmse': rmse, 'n_pontos': len(n), 'melhor': True})
    _plot_bnb(rows, pasta_fig)


def main(perfil_csv, bnb_csv, saida):
    linhas = []
    pasta_fig = os.path.join(os.path.dirname(saida) or '.', 'figuras')
    parte_aproximados(perfil_csv, linhas, pasta_fig)
    parte_bnb(bnb_csv, linhas, pasta_fig)
    if linhas:
        os.makedirs(os.path.dirname(saida) or '.', exist_ok=True)
        with open(saida, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)
        print(f"\nAjustes salvos em: {saida} | figuras em: {pasta_fig}/")


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else 'results/profiling_subrotinas.csv',
        sys.argv[2] if len(sys.argv) > 2 else 'results/resultados_bnb.csv',
        sys.argv[3] if len(sys.argv) > 3 else 'results/regressao_complexidade.csv',
    )
import csv
import math
import os
import sys

import numpy as np
from scipy import stats

try:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor, export_text
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

FEATURES_MST = [
    'n', 'cv_nn', 'grau_max_mst', 'frac_folhas_mst',
    'frac_impares_mst', 'cv_arestas_mst', 'razao_mst_max',
]

FEATURES_GEO = [
    'razao_aspecto', 'frac_hull', 'entropia_espacial', 'clark_evans',
]

ALVOS_GAP = [
    ('gap_twice_pct', 'Twice-around-the-tree'),
    ('gap_christofides_pct', 'Christofides'),
    ('gap_nn_pct', 'Nearest Neighbor'),
    ('gap_hull_pct', 'Convex Hull Insertion'),
]


def _num(v):
    if v in (None, '', 'NA', 'na', 'nan', 'NaN'):
        return None
    try:
        x = float(v)
        return None if math.isnan(x) or math.isinf(x) else x
    except ValueError:
        return None


def _ler_csv(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    with open(caminho, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def carregar_dataset(features_csv='results/features.csv',
                     gaps_csv='results/gaps.csv',
                     bnb_csv='results/resultados_bnb.csv'):
    feats = {r['instancia']: r for r in _ler_csv(features_csv)}
    gaps = {r['instancia']: r for r in _ler_csv(gaps_csv)}

    bnb_por_inst = {}
    if os.path.exists(bnb_csv):
        for r in _ler_csv(bnb_csv):
            if r.get('estrategia') == 'christofides' and r.get('status') == 'OK':
                bnb_por_inst[r['instancia']] = r

    dados = []
    for nome, fr in feats.items():
        if nome not in gaps:
            continue
        gr = gaps[nome]
        linha = {'instancia': nome, 'tipo': fr.get('tipo', ''), 'faixa': fr.get('faixa', '')}
        for col in FEATURES_MST + FEATURES_GEO:
            linha[col] = _num(fr.get(col))
        for col, _ in ALVOS_GAP:
            linha[col] = _num(gr.get(col))
        br = bnb_por_inst.get(nome)
        linha['bnb_nos_exp_chris'] = _num(br.get('nos_expandidos')) if br else None
        linha['bnb_t_ms_chris'] = _num(br.get('t_bnb_ms')) if br else None
        dados.append(linha)

    return dados


def _filtrar_matriz(dados, cols_feat, col_alvo):
    linhas_ok = [
        r for r in dados
        if r.get(col_alvo) is not None and all(r.get(c) is not None for c in cols_feat)
    ]
    if not linhas_ok:
        return [], np.empty((0, 0)), np.empty(0), []

    X = np.array([[r[c] for c in cols_feat] for r in linhas_ok], dtype=float)
    y = np.array([r[col_alvo] for r in linhas_ok], dtype=float)
    nomes_inst = [r['instancia'] for r in linhas_ok]

    # remove colunas constantes (std == 0) para evitar NaN na correlação/padronização
    mask_var = X.std(axis=0) > 1e-12
    cols_ativas = [c for c, ok in zip(cols_feat, mask_var) if ok]
    X = X[:, mask_var]
    return nomes_inst, X, y, cols_ativas


def calcular_correlacoes(dados, grupos, alvos):
    linhas = []
    for nome_grupo, cols in grupos:
        for col_alvo, rot_alvo in alvos:
            nomes, X, y, cols_ativas = _filtrar_matriz(dados, cols, col_alvo)
            if len(nomes) < 4 or y.std() < 1e-12:
                continue
            for j, feat in enumerate(cols_ativas):
                xj = X[:, j]
                r_p, p_p = stats.pearsonr(xj, y)
                r_s, p_s = stats.spearmanr(xj, y)
                linhas.append({
                    'grupo_features': nome_grupo,
                    'alvo': col_alvo,
                    'rotulo_alvo': rot_alvo,
                    'feature': feat,
                    'n_instancias': len(nomes),
                    'pearson_r': round(float(r_p), 4),
                    'pearson_p': round(float(p_p), 4),
                    'spearman_rho': round(float(r_s), 4),
                    'spearman_p': round(float(p_s), 4),
                })
    return linhas


def ajustar_modelos(dados, grupos, alvos):
    if not SKLEARN_OK:
        print("AVISO: scikit-learn não instalado; instale com `pip install scikit-learn` "
              "para rodar regressão múltipla e árvore de decisão.")
        return [], []

    linhas_pesos = []
    textos = []

    for nome_grupo, cols in grupos:
        for col_alvo, rot_alvo in alvos:
            nomes, X, y, cols_ativas = _filtrar_matriz(dados, cols, col_alvo)
            m = len(nomes)
            if m < 6 or len(cols_ativas) == 0 or y.std() < 1e-12:
                continue

            scaler = StandardScaler()
            Xz = scaler.fit_transform(X)

            # Regressão Múltipla Padronizada (RidgeCV para estabilidade com N pequeno)
            ridge = RidgeCV(alphas=np.logspace(-2, 2, 25))
            ridge.fit(Xz, y)
            r2_treino = float(ridge.score(Xz, y))

            # Validação Leave-One-Out (LOOCV)
            loo = LeaveOneOut()
            y_pred_loo = np.zeros(m)
            for tr, te in loo.split(X):
                sc_i = StandardScaler()
                Xtr = sc_i.fit_transform(X[tr])
                Xte = sc_i.transform(X[te])
                mod_i = RidgeCV(alphas=np.logspace(-2, 2, 25))
                mod_i.fit(Xtr, y[tr])
                y_pred_loo[te] = mod_i.predict(Xte)

            ss_res = float(((y - y_pred_loo) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2_loocv = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
            rmse_loocv = math.sqrt(ss_res / m)

            # Árvore de Decisão rasa para regras interpretáveis
            arvore = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=42)
            arvore.fit(X, y)
            r2_arv = float(arvore.score(X, y))
            regras = export_text(arvore, feature_names=cols_ativas)

            cabecalho = (
                f"\n[{nome_grupo}] Alvo: {rot_alvo} ({col_alvo}) | n_inst={m}\n"
                f"  Ridge (alpha={ridge.alpha_:.3g}): R2_treino={r2_treino:.3f} | "
                f"R2_LOOCV={r2_loocv:.3f} | RMSE_LOOCV={rmse_loocv:.2f} pp\n"
                f"  Árvore (max_depth=2): R2_treino={r2_arv:.3f}\n"
                f"{regras}"
            )
            textos.append(cabecalho)

            for feat, beta, imp in zip(cols_ativas, ridge.coef_, arvore.feature_importances_):
                linhas_pesos.append({
                    'grupo_features': nome_grupo,
                    'alvo': col_alvo,
                    'n_instancias': m,
                    'feature': feat,
                    'beta_padronizado': round(float(beta), 4),
                    'abs_beta': round(float(abs(beta)), 4),
                    'importancia_arvore': round(float(imp), 4),
                    'r2_ridge_treino': round(r2_treino, 4),
                    'r2_ridge_loocv': round(r2_loocv, 4),
                    'rmse_ridge_loocv': round(rmse_loocv, 4),
                    'r2_arvore_treino': round(r2_arv, 4),
                })

    return linhas_pesos, textos


def analisar_adversariais(dados, alvos):
    todas_feats = FEATURES_MST + FEATURES_GEO
    medias, desvios = {}, {}
    for f in todas_feats:
        vals = [r[f] for r in dados if r.get(f) is not None]
        if len(vals) >= 2:
            arr = np.array(vals, float)
            medias[f] = float(arr.mean())
            desvios[f] = float(arr.std()) if arr.std() > 1e-12 else 0.0

    linhas = ["\n=== Instâncias Adversariais (Top-3 maiores gaps e features extremas) ==="]
    for col_alvo, rot_alvo in alvos:
        validas = [r for r in dados if r.get(col_alvo) is not None]
        if not validas:
            continue
        top = sorted(validas, key=lambda r: r[col_alvo], reverse=True)[:3]
        linhas.append(f"\n{rot_alvo} ({col_alvo}):")
        for r in top:
            zs = []
            for f in todas_feats:
                v = r.get(f)
                if v is not None and desvios.get(f, 0.0) > 0:
                    z = (v - medias[f]) / desvios[f]
                    if abs(z) >= 0.8:
                        zs.append((abs(z), f, v, z))
            zs.sort(reverse=True)
            dest = ", ".join(f"{f}={v:.3g} (z={z:+.2f})" for _, f, v, z in zs[:4])
            linhas.append(f"  - {r['instancia']:<13} n={int(r['n']):<2} gap={r[col_alvo]:>6.2f}% | {dest or 'sem desvios extremos'}")
    return "\n".join(linhas)


def gerar_figuras(dados, correlacoes, pesos, pasta_fig='results/figuras'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(pasta_fig, exist_ok=True)

    # 1) Heatmap de Spearman (MST + Geo quando disponível)
    alvos_plot = [c for c, _ in ALVOS_GAP[:3]]
    feats_plot = FEATURES_MST + FEATURES_GEO
    M = np.full((len(feats_plot), len(alvos_plot)), np.nan)
    for j, alvo in enumerate(alvos_plot):
        grupo = 'GEO_E_MST' if any(
            c['alvo'] == alvo and c['grupo_features'] == 'GEO_E_MST' for c in correlacoes
        ) else 'MST'
        for i, feat in enumerate(feats_plot):
            g_use = 'GEO_E_MST' if feat in FEATURES_GEO else 'MST'
            achou = [
                c for c in correlacoes
                if c['alvo'] == alvo and c['feature'] == feat and c['grupo_features'] == g_use
            ]
            if not achou and grupo != g_use:
                achou = [c for c in correlacoes if c['alvo'] == alvo and c['feature'] == feat]
            if achou:
                M[i, j] = achou[0]['spearman_rho']

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(np.nan_to_num(M, nan=0.0), cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(alvos_plot)))
    ax.set_xticklabels(['Twice-around', 'Christofides', 'Nearest Neighbor'], rotation=15)
    ax.set_yticks(range(len(feats_plot)))
    ax.set_yticklabels(feats_plot, fontsize=8)
    for i in range(len(feats_plot)):
        for j in range(len(alvos_plot)):
            val = M[i, j]
            txt = 'NA' if math.isnan(val) else f"{val:+.2f}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='black')
    fig.colorbar(im, ax=ax, label='Correlação de Spearman (ρ)')
    ax.set_title('Sensibilidade: Correlação Features × Gap (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, 'sensibilidade_correlacoes.png'), dpi=150)
    plt.close(fig)

    # 2) Barras com pesos padronizados Ridge (grupo MST para cobrir todas as instâncias)
    if pesos:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, (alvo, tit) in zip(axes, [('gap_twice_pct', 'Twice-around'),
                                          ('gap_christofides_pct', 'Christofides')]):
            sub = [p for p in pesos if p['grupo_features'] == 'MST' and p['alvo'] == alvo]
            if not sub:
                continue
            sub.sort(key=lambda p: p['beta_padronizado'])
            nomes_f = [p['feature'] for p in sub]
            vals_b = [p['beta_padronizado'] for p in sub]
            cores = ['#c0392b' if v > 0 else '#2980b9' for v in vals_b]
            ax.barh(nomes_f, vals_b, color=cores)
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_title(f"{tit}\n(R²_treino={sub[0]['r2_ridge_treino']:.2f}, RMSE_LOO={sub[0]['rmse_ridge_loocv']:.1f}pp)", fontsize=9)
            ax.set_xlabel('Peso padronizado β (Ridge)')
        fig.tight_layout()
        fig.savefig(os.path.join(pasta_fig, 'sensibilidade_pesos_ridge.png'), dpi=150)
        plt.close(fig)

    # 3) Dispersão dos principais sinais (cv_nn e frac_impares_mst vs gaps)
    pares = [
        ('cv_nn', 'gap_nn_pct', 'CV distância NN vs Gap Nearest Neighbor'),
        ('frac_impares_mst', 'gap_twice_pct', 'Fração vértices ímpares MST vs Gap Twice-around'),
        ('frac_hull', 'gap_christofides_pct', 'Fração no Fecho Convexo vs Gap Christofides'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (fx, fy, tit) in zip(axes, pares):
        pts = [r for r in dados if r.get(fx) is not None and r.get(fy) is not None]
        if not pts:
            continue
        xs = [r[fx] for r in pts]
        ys = [r[fy] for r in pts]
        ax.scatter(xs, ys, color='#2c3e50')
        for r in pts:
            ax.annotate(r['instancia'], (r[fx], r[fy]), fontsize=6.5,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel(fx)
        ax.set_ylabel(fy)
        ax.set_title(tit, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(pasta_fig, 'sensibilidade_dispersao.png'), dpi=150)
    plt.close(fig)


def main(features_csv='results/features.csv',
         gaps_csv='results/gaps.csv',
         bnb_csv='results/resultados_bnb.csv',
         pasta_saida='results'):
    os.makedirs(pasta_saida, exist_ok=True)
    dados = carregar_dataset(features_csv, gaps_csv, bnb_csv)
    if not dados:
        print("Nenhum dado combinado encontrado. Rode features.py e gaps.py antes.")
        return

    grupos = [
        ('MST', FEATURES_MST),
        ('GEO_E_MST', FEATURES_MST + FEATURES_GEO),
    ]

    correlacoes = calcular_correlacoes(dados, grupos, ALVOS_GAP)
    pesos, textos_modelos = ajustar_modelos(dados, grupos, ALVOS_GAP)
    texto_adv = analisar_adversariais(dados, ALVOS_GAP)

    corr_csv = os.path.join(pasta_saida, 'sensibilidade_correlacoes.csv')
    if correlacoes:
        with open(corr_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(correlacoes[0].keys()))
            w.writeheader()
            w.writerows(correlacoes)

    pesos_csv = os.path.join(pasta_saida, 'sensibilidade_modelos.csv')
    if pesos:
        with open(pesos_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(pesos[0].keys()))
            w.writeheader()
            w.writerows(pesos)

    rel_txt = os.path.join(pasta_saida, 'sensibilidade_relatorio.txt')
    resumo = []
    resumo.append("=== Top Correlações (|Spearman| >= 0.40) ===")
    for c in sorted(correlacoes, key=lambda x: abs(x['spearman_rho']), reverse=True):
        if abs(c['spearman_rho']) >= 0.40:
            resumo.append(
                f"  [{c['grupo_features']:<9}] {c['alvo']:<21} ~ {c['feature']:<18} "
                f"rho={c['spearman_rho']:+.3f} (p={c['spearman_p']:.3f}) | "
                f"r={c['pearson_r']:+.3f} (n={c['n_instancias']})"
            )
    resumo.extend(textos_modelos)
    resumo.append(texto_adv)

    conteudo = "\n".join(resumo)
    with open(rel_txt, 'w', encoding='utf-8') as f:
        f.write(conteudo)

    gerar_figuras(dados, correlacoes, pesos, os.path.join(pasta_saida, 'figuras'))

    print(conteudo)
    print(f"\nArquivos salvos em: {corr_csv}, {pesos_csv}, {rel_txt} e {pasta_saida}/figuras/")


if __name__ == '__main__':
    main(
        sys.argv[1] if len(sys.argv) > 1 else 'results/features.csv',
        sys.argv[2] if len(sys.argv) > 2 else 'results/gaps.csv',
        sys.argv[3] if len(sys.argv) > 3 else 'results/resultados_bnb.csv',
    )
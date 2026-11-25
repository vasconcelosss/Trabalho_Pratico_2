import os
import sys

import numpy as np
from flask import Flask, jsonify, render_template_string, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from branch_and_bound import branch_and_bound
from christofides import _circuito_euleriano
from features import _projetar
from heuristics import findConvexHull_indices, nearest_neighbor
from instancias import INSTANCIAS, _achar_arquivo
from matching import emparelhamento_perfeito_minimo
from mst import adjacencia_mst, prim, vertices_grau_impar
from utils import (calcular_custo_rota, calcular_matriz_distancias,
                   ler_instancia_tsplib_completa)

app = Flask(__name__)

HTML = r"""
<!doctype html>
<html lang="pt-br">
<head>
  <meta charset="utf-8">
  <title>TSP Visualizer — DCC207</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #f6f7fb; color: #1f2937; }
    header { background: #1e293b; color: #fff; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; }
    header h1 { margin: 0; font-size: 18px; }
    .wrap { max-width: 1120px; margin: 20px auto; padding: 0 16px; display: grid; grid-template-columns: 320px 1fr; gap: 18px; }
    .card { background: #fff; border-radius: 10px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
    label { display: block; font-size: 12px; font-weight: 600; margin-top: 10px; color: #475569; }
    select, button { width: 100%; padding: 8px 10px; margin-top: 5px; border-radius: 6px; border: 1px solid #cbd5e1; font-size: 14px; }
    button { background: #2563eb; color: #fff; border: none; font-weight: 600; cursor: pointer; margin-top: 12px; }
    button:hover { background: #1d4ed8; }
    .row-btns { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 10px; }
    .row-btns button { margin-top: 0; background: #334155; }
    .row-btns button.play { background: #059669; }
    .info { margin-top: 14px; font-size: 13px; line-height: 1.5; background: #f8fafc; padding: 10px; border-radius: 8px; border: 1px solid #e2e8f0; }
    .legend { font-size: 12px; margin-top: 12px; display: grid; gap: 5px; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
    canvas { width: 100%; height: 600px; background: #fff; border-radius: 10px; border: 1px solid #e2e8f0; }
    .step-title { font-size: 15px; font-weight: 600; margin-bottom: 8px; min-height: 22px; }
  </style>
</head>
<body>
<header>
  <h1>Visualizador Passo a Passo — TSP (B&amp;B, Twice-around, Christofides &amp; Heurísticas)</h1>
  <span style="font-size:12px;opacity:.8">DCC207 • Algoritmos 2</span>
</header>
<div class="wrap">
  <div class="card">
    <label>Instância TSPLIB</label>
    <select id="instancia"></select>

    <label>Algoritmo</label>
    <select id="algoritmo">
      <option value="christofides">Christofides (3/2-aprox)</option>
      <option value="twice_around">Twice-around-the-tree (2-aprox)</option>
      <option value="convex_hull">Convex Hull Insertion</option>
      <option value="nearest_neighbor">Nearest Neighbor</option>
      <option value="bnb">Branch-and-Bound (Exato, timeout 5s)</option>
    </select>

    <button onclick="carregarExecucao()">Executar / Gerar Passos</button>

    <div class="row-btns">
      <button onclick="mudarPasso(-1)">◀ Anterior</button>
      <button class="play" id="btnPlay" onclick="togglePlay()">▶ Play</button>
      <button onclick="mudarPasso(1)">Próximo ▶</button>
    </div>

    <label>Velocidade (<span id="velLabel">350</span> ms)</label>
    <input type="range" id="vel" min="60" max="1200" value="350" style="width:100%"
           oninput="document.getElementById('velLabel').textContent=this.value">

    <div class="info" id="meta">Selecione uma instância e clique em Executar.</div>

    <div class="legend">
      <div><span class="dot" style="background:#2563eb"></span>Arestas da MST (Prim)</div>
      <div><span class="dot" style="background:#ef4444"></span>Vértices de grau ímpar / destaque</div>
      <div><span class="dot" style="background:#f59e0b"></span>Matching Perfeito Mínimo</div>
      <div><span class="dot" style="background:#10b981"></span>Rota / Sub-rota construída</div>
    </div>
  </div>

  <div class="card">
    <div class="step-title" id="stepTitle">—</div>
    <canvas id="canvas" width="760" height="600"></canvas>
  </div>
</div>

<script>
let dados = null;
let idx = 0;
let timer = null;

async function init() {
  const res = await fetch('/api/instancias');
  const lista = await res.json();
  const sel = document.getElementById('instancia');
  lista.forEach(it => {
    const opt = document.createElement('option');
    opt.value = it.nome;
    opt.textContent = `${it.nome} (n=${it.n}, ${it.tipo})`;
    sel.appendChild(opt);
  });
  sel.value = lista.some(x => x.nome === 'ulysses16') ? 'ulysses16' : lista[0].nome;
  carregarExecucao();
}

async function carregarExecucao() {
  pararPlay();
  const inst = document.getElementById('instancia').value;
  const alg = document.getElementById('algoritmo').value;
  const res = await fetch(`/api/executar?instancia=${encodeURIComponent(inst)}&algoritmo=${alg}`);
  dados = await res.json();
  idx = 0;
  const gapTxt = dados.gap_pct !== null ? `${dados.gap_pct.toFixed(2)}%` : 'NA';
  const otTxt = dados.otimo !== null ? dados.otimo : 'NA';
  document.getElementById('meta').innerHTML =
    `<b>Instância:</b> ${dados.nome} (n=${dados.n}, ${dados.tipo})<br>` +
    `<b>Coords:</b> ${dados.layout_msg}<br>` +
    `<b>Ótimo tabela:</b> ${otTxt}<br>` +
    `<b>Custo final:</b> ${dados.custo ?? 'TIMEOUT'}<br>` +
    `<b>Gap:</b> ${gapTxt}<br>` +
    `<b>Passos:</b> ${dados.steps.length}`;
  desenhar();
}

function mudarPasso(delta) {
  if (!dados) return;
  idx = Math.max(0, Math.min(dados.steps.length - 1, idx + delta));
  desenhar();
}

function pararPlay() {
  if (timer) { clearInterval(timer); timer = null; }
  document.getElementById('btnPlay').textContent = '▶ Play';
}

function togglePlay() {
  if (!dados) return;
  if (timer) { pararPlay(); return; }
  document.getElementById('btnPlay').textContent = '❚❚ Pausar';
  const ms = parseInt(document.getElementById('vel').value, 10);
  timer = setInterval(() => {
    if (idx >= dados.steps.length - 1) { pararPlay(); return; }
    idx++;
    desenhar();
  }, ms);
}

function desenhar() {
  if (!dados) return;
  const step = dados.steps[idx];
  document.getElementById('stepTitle').textContent =
    `Passo ${idx + 1}/${dados.steps.length}: ${step.titulo}`;

  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height, pad = 42;
  ctx.clearRect(0, 0, W, H);

  const pts = dados.pts;
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const scale = Math.min((W - 2 * pad) / (maxX - minX || 1), (H - 2 * pad) / (maxY - minY || 1));
  const offX = (W - (maxX - minX) * scale) / 2;
  const offY = (H - (maxY - minY) * scale) / 2;
  const map = ([x, y]) => [offX + (x - minX) * scale, H - (offY + (y - minY) * scale)];

  // MST
  ctx.strokeStyle = '#2563eb';
  ctx.lineWidth = 2;
  ctx.setLineDash([]);
  (step.mst_edges || []).forEach(([u, v]) => {
    const [x1, y1] = map(pts[u]), [x2, y2] = map(pts[v]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  });

  // Matching
  ctx.strokeStyle = '#f59e0b';
  ctx.lineWidth = 2.2;
  ctx.setLineDash([7, 5]);
  (step.match_edges || []).forEach(([u, v]) => {
    const [x1, y1] = map(pts[u]), [x2, y2] = map(pts[v]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  });

  // Rota / caminho
  ctx.setLineDash([]);
  const path = step.path || [];
  if (path.length >= 2) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2.8;
    ctx.beginPath();
    const [sx, sy] = map(pts[path[0]]);
    ctx.moveTo(sx, sy);
    for (let i = 1; i < path.length; i++) {
      const [px, py] = map(pts[path[i]]);
      ctx.lineTo(px, py);
    }
    if (step.closed) ctx.lineTo(sx, sy);
    ctx.stroke();
  }

  // Vértices
  const hi = new Set(step.highlight_nodes || []);
  pts.forEach((p, i) => {
    const [x, y] = map(p);
    ctx.beginPath();
    ctx.arc(x, y, hi.has(i) ? 7.5 : 5, 0, Math.PI * 2);
    ctx.fillStyle = hi.has(i) ? '#ef4444' : '#1e293b';
    ctx.fill();
    ctx.fillStyle = '#334155';
    ctx.font = '11px system-ui';
    ctx.fillText(String(i), x + 7, y - 6);
  });
}

init();
</script>
</body>
</html>
"""


def _mds_2d(distancias, n):
    """Projeção 2D clássica (MDS) caso a instância EXPLICIT não tenha coords."""
    D2 = np.array(distancias, dtype=float) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:2]
    vals = np.maximum(vals[idx], 0.0)
    return vecs[:, idx] * np.sqrt(vals)


def _carregar_uma(nome, pasta='data'):
    entrada = next((e for e in INSTANCIAS if e['nome'] == nome), None)
    base = entrada.get('base', nome) if entrada else nome
    d = ler_instancia_tsplib_completa(_achar_arquivo(pasta, base))
    if entrada and 'k' in entrada:
        k = entrada['k']
        coords = {i: d['coords'][i] for i in range(k)}
        dist = calcular_matriz_distancias(coords, k, d['tipo'])
        n = k
    else:
        coords, dist, n = d['coords'], d['distancias'], d['n']

    if coords and len(coords) >= n:
        P = _projetar(coords, n, d['tipo'])
        layout_msg = 'Coordenadas reais/display TSPLIB'
    else:
        P = _mds_2d(dist, n)
        layout_msg = 'Projeção MDS 2D (instância EXPLICIT sem coords)'

    otimo = entrada['otimo'] if entrada else None
    return d['tipo'], n, coords, dist, P, otimo, layout_msg


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/instancias')
def listar_instancias():
    saida = []
    for e in INSTANCIAS:
        base = e.get('base', e['nome'])
        try:
            d = ler_instancia_tsplib_completa(_achar_arquivo('data', base))
            n = e.get('k', d['n'])
            saida.append({'nome': e['nome'], 'n': n, 'tipo': d['tipo']})
        except FileNotFoundError:
            continue
    return jsonify(saida)


@app.route('/api/executar')
def executar():
    nome = request.args.get('instancia', 'ulysses16')
    alg = request.args.get('algoritmo', 'christofides')
    tipo, n, coords, dist, P, otimo, layout_msg = _carregar_uma(nome)
    steps = []
    custo = None

    if alg == 'christofides':
        arestas_mst = prim(dist, n)
        steps.append({'titulo': 'Pontos da instância', 'mst_edges': [], 'match_edges': [], 'path': []})
        mst_pares = []
        for u, v, _ in arestas_mst:
            mst_pares.append([u, v])
            steps.append({'titulo': f'Prim: adicionando aresta ({u}, {v}) à MST',
                          'mst_edges': list(mst_pares), 'highlight_nodes': [u, v]})
        impares = vertices_grau_impar(arestas_mst, n)
        steps.append({'titulo': f'{len(impares)} vértices de grau ímpar identificados na MST',
                      'mst_edges': list(mst_pares), 'highlight_nodes': impares})
        arestas_match, _ = emparelhamento_perfeito_minimo(dist, n, arestas_mst)
        match_pares = [[u, v] for u, v in arestas_match]
        steps.append({'titulo': 'Emparelhamento perfeito mínimo (Blossom) nos vértices ímpares',
                      'mst_edges': list(mst_pares), 'match_edges': match_pares,
                      'highlight_nodes': impares})

        adj_multi = {v: [] for v in range(n)}
        for u, v, _ in arestas_mst:
            adj_multi[u].append(v)
            adj_multi[v].append(u)
        for u, v in arestas_match:
            adj_multi[u].append(v)
            adj_multi[v].append(u)
        circuito = _circuito_euleriano(adj_multi, n)

        visitados = set()
        rota = []
        for v in circuito:
            if v not in visitados:
                visitados.add(v)
                rota.append(v)
                steps.append({'titulo': f'Shortcut do circuito euleriano: visitando vértice {v}',
                              'mst_edges': list(mst_pares), 'match_edges': match_pares,
                              'path': list(rota), 'highlight_nodes': [v], 'closed': False})
        custo = calcular_custo_rota(rota, dist)
        steps.append({'titulo': f'Rota final Christofides (custo = {custo})',
                      'path': rota, 'closed': True})

    elif alg == 'twice_around':
        arestas_mst = prim(dist, n)
        mst_pares = [[u, v] for u, v, _ in arestas_mst]
        steps.append({'titulo': 'MST construída via Prim', 'mst_edges': mst_pares})
        adj = adjacencia_mst(arestas_mst, n)
        rota, visitados = [], set()

        def dfs_anim(v):
            visitados.add(v)
            rota.append(v)
            steps.append({'titulo': f'DFS em pré-ordem (shortcut): visitando {v}',
                          'mst_edges': mst_pares, 'path': list(rota),
                          'highlight_nodes': [v], 'closed': False})
            for viz, _ in adj[v]:
                if viz not in visitados:
                    dfs_anim(viz)

        dfs_anim(0)
        custo = calcular_custo_rota(rota, dist)
        steps.append({'titulo': f'Rota final Twice-around (custo = {custo})',
                      'path': rota, 'closed': True})

    elif alg == 'nearest_neighbor':
        visitados = [False] * n
        visitados[0] = True
        rota = [0]
        atual = 0
        steps.append({'titulo': 'Início no vértice 0', 'path': [0], 'highlight_nodes': [0]})
        for _ in range(n - 1):
            prox = min((v for v in range(n) if not visitados[v]), key=lambda v: dist[atual][v])
            visitados[prox] = True
            rota.append(prox)
            atual = prox
            steps.append({'titulo': f'Vizinho mais próximo: indo para {prox}',
                          'path': list(rota), 'highlight_nodes': [prox], 'closed': False})
        custo = calcular_custo_rota(rota, dist)
        steps.append({'titulo': f'Rota final Nearest Neighbor (custo = {custo})',
                      'path': rota, 'closed': True})

    elif alg == 'convex_hull':
        coords_2d = {i: (float(P[i, 0]), float(P[i, 1])) for i in range(n)}
        sub_rota = findConvexHull_indices(coords_2d, n)
        steps.append({'titulo': f'Fecho Convexo inicial ({len(sub_rota)} vértices)',
                      'path': list(sub_rota), 'highlight_nodes': list(sub_rota), 'closed': True})
        na_rota = set(sub_rota)
        internos = [v for v in range(n) if v not in na_rota]
        while internos:
            melhor_v, melhor_pos, melhor_delta = -1, -1, float('inf')
            m = len(sub_rota)
            for v in internos:
                for i in range(m):
                    u, w = sub_rota[i], sub_rota[(i + 1) % m]
                    delta = dist[u][v] + dist[v][w] - dist[u][w]
                    if delta < melhor_delta:
                        melhor_delta, melhor_v, melhor_pos = delta, v, i + 1
            sub_rota.insert(melhor_pos, melhor_v)
            internos.remove(melhor_v)
            steps.append({'titulo': f'Inserindo vértice interno {melhor_v} (Δ = {melhor_delta})',
                          'path': list(sub_rota), 'highlight_nodes': [melhor_v], 'closed': True})
        custo = calcular_custo_rota(sub_rota, dist)
        steps.append({'titulo': f'Rota final Convex Hull Insertion (custo = {custo})',
                      'path': sub_rota, 'closed': True})

    elif alg == 'bnb':
        _, c_init = nearest_neighbor(dist, n)
        res = branch_and_bound(dist, n, timeout=5.0)
        rota = res['melhor_rota_parcial'] or list(range(n))
        custo = res['custo'] if res['status'] == 'OK' else res['melhor_custo_parcial']
        steps.append({'titulo': f"B&B ({res['status']}) | exp={res['nos_expandidos']} pod={res['nos_podados']} | custo={custo}",
                      'path': rota, 'closed': True})

    gap_pct = ((custo - otimo) / otimo * 100.0) if (custo is not None and otimo) else None
    return jsonify({
        'nome': nome, 'n': n, 'tipo': tipo, 'otimo': otimo,
        'layout_msg': layout_msg,
        'pts': P.tolist(), 'steps': steps, 'custo': custo, 'gap_pct': gap_pct,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
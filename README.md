# Trabalho Prático 2 — Análise Empírica e Otimização para o Caixeiro Viajante (TSP)

**Disciplina:** DCC207 — Algoritmos 2 (UFMG)  
**Tema:** Auditoria, *profiling*, modelagem empírica de complexidade e análise de sensibilidade para o Problema do Caixeiro Viajante Simétrico com:
- **Branch-and-Bound (Exato)** com 5 estratégias de *warm-start* (`cold_start`, `nearest_neighbor`, `convex_hull`, `twice_around`, `christofides`), quebra de simetria e potenciais Lagrangeanos de Held-Karp;
- **Twice-around-the-tree** (aproximação fator 2);
- **Christofides** (aproximação fator 3/2, via Prim + Blossom/NetworkX + Hierholzer).

---

## Estrutura do Projeto

```bash
.
├── app.py                  # Visualizador web passo a passo (Flask + Canvas)
├── main.py                 # Ponto de entrada (instância única ou experimento B&B)
├── Makefile                # Atalhos para todo o pipeline experimental
├── requirements.txt        # Dependências Python
├── data/                   # Instâncias TSPLIB (.tsp / .tsp.gz)
├── results/                # CSVs, logs, cProfile e figuras geradas
├── src/
│   ├── branch_and_bound.py # B&B (DFS / Best-First, Held-Karp, warm-starts)
│   ├── christofides.py     # Algoritmo de Christofides instrumentado por etapa
│   ├── twice_around.py     # Twice-around-the-tree
│   ├── heuristics.py       # Cold start, Nearest Neighbor e Convex Hull Insertion
│   ├── mst.py              # Árvore Geradora Mínima (Prim) e utilitários de grau
│   ├── matching.py         # Emparelhamento perfeito mínimo (NetworkX)
│   ├── medicao.py          # Medição isolada de tempo (mediana) e memória (tracemalloc)
│   ├── instancias.py       # Conjunto de teste TSPLIB + validação
│   ├── experimento_bnb.py  # Executor do B&B com warm-starts e exportação CSV
│   ├── features.py         # Extração de métricas geométricas e topológicas da MST
│   ├── gaps.py             # Conferência dos ótimos e cálculo dos gaps (%)
│   ├── profilling_for_christofides.py # Profiling por subrotina + GC + cProfile
│   ├── regression.py       # Ajuste assintótico (R², RMSE) e curvas de complexidade
│   ├── sensibilidade.py    # Correlações, Ridge, Árvore de Decisão e adversariais
│   ├── plot.py             # Gráficos comparativos de rotas/MST/matching
│   └── utils.py            # Leitura TSPLIB95 (EUC_2D, CEIL_2D, GEO, ATT, EXPLICIT)
└── tests/                  # Testes unitários baseados em assert
```

---

## Pré-requisitos e Instalação

Recomenda-se Python 3.10+ dentro de um ambiente virtual (`venv`):

```bash
python3 -m venv venv
source venv/bin/activate
make install
```

Ou diretamente com `pip`:

```bash
pip install -r requirements.txt
```

> Certifique-se de que os arquivos `.tsp` (ou `.tsp.gz`) da TSPLIB estejam na pasta `data/`.

---

## Como Rodar com `Makefile`

### 1. Validar instâncias e rodar testes
```bash
make validate      # Confere leitura TSPLIB, simetria e sanidade das rotas
make test          # Executa todos os scripts de verificação em tests/
```

### 2. Executar o pipeline completo
- **Modo oficial (limite de 30 min = `1800s` por execução do B&B):**
  ```bash
  make all
  ```
- **Modo rápido de validação (`60s` de timeout no B&B):**
  ```bash
  make all-fast
  ```

### 3. Executar etapas individualmente
```bash
make features      # Extrai features -> results/features.csv
make profiling     # Profiling de subrotinas + cProfile
make bnb           # B&B oficial (TIMEOUT=1800s)
make bnb-fast      # B&B rápido (TIMEOUT=60s)
make bnb-bg        # B&B oficial em segundo plano (nohup -> results/log_bnb.txt)
make post          # Roda gaps + regression + sensibilidade + plots
```

Se rodar em segundo plano com `make bnb-bg`, acompanhe o progresso com:
```bash
tail -f results/log_bnb.txt
```
Quando terminar, execute `make post` para consolidar todas as análises e figuras.

### 4. Visualizador Web Passo a Passo (Flask)
Para visualizar no navegador a construção da MST, identificação de vértices ímpares, *matching*, circuito euleriano/atalhos e rotas finais:

```bash
make web
```
Acesse **`http://127.0.0.1:5000`** no navegador.

---

## Como Rodar Diretamente com Python (sem `make`)

### Demonstração rápida (instância única)
```bash
python3 main.py                  # Instância embutida (pentágono de 5 vértices)
python3 main.py data/gr24.tsp    # Profiling e rotas para uma instância TSPLIB
```

### Pipeline completo passo a passo
```bash
# 1. Validação das instâncias e testes
python3 src/instancias.py data

# 2. Extração de features e profiling dos aproximados
python3 src/features.py data results/features.csv
python3 src/profilling_for_christofides.py data 7 5

# 3. Experimento Branch-and-Bound (timeout em segundos: 1800 = 30 min)
python3 main.py bnb data 1800

# 4. Pós-processamento: gaps, regressão, sensibilidade e grafos
python3 src/gaps.py
python3 src/regression.py
python3 src/sensibilidade.py
python3 src/plot.py ulysses16 ulysses22 bays29 berlin52
```

### Visualizador Web e Testes
```bash
python3 app.py
python3 tests/<arquivo_de_teste>.py
```

---

## Arquivos Gerados em `results/`

| Arquivo | Descrição |
|---|---|
| `resultados_bnb.csv` | Métricas por instância e warm-start (nós expandidos, podados, profundidade, tempo e memória; `NA` em timeouts) |
| `profiling_subrotinas.csv` | Tempo mediano por etapa do Christofides (com e sem GC), Twice-around e picos de memória |
| `cprofile_christofides.txt` | Perfil detalhado (`cProfile`) na maior instância |
| `regressao_complexidade.csv` | Ajustes assintóticos ($N$, $N\log N$, $N^2$, $N^3$, $N^b$, $e^{bN}$) com $R^2$ e RMSE |
| `features.csv` | Métricas geométricas (aspecto, fecho convexo, entropia, Clark-Evans) e topológicas da MST |
| `gaps.csv` | Conferência dos ótimos (B&B × TSPLIB) e *gaps* percentuais de todas as heurísticas |
| `sensibilidade_correlacoes.csv` | Correlações de Pearson e Spearman entre *features* e *gaps* |
| `sensibilidade_modelos.csv` | Pesos padronizados (`Ridge`), validação LOOCV e importâncias da Árvore de Decisão |
| `sensibilidade_relatorio.txt` | Resumo textual das regras da árvore e instâncias adversariais |
| `figuras/` | Gráficos de regressão, sensibilidade e painéis comparativos de grafos (`figuras/grafos/`) |
```
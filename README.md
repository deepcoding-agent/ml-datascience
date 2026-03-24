# ml-datascience — PrepPilot FastAPI Backend

FastAPI service powering the PrepPilot AI data science platform. Features an **AI-first agent architecture** where an LLM planner is the sole decision-maker — no hardcoded keywords or regex routing. 101 pre-built handlers, LLM-powered code generation fallback, and sandboxed Python execution. Supports **Anthropic Claude** and **OpenAI GPT** models with automatic provider detection. Returns text answers, interactive Plotly charts, cleaned datasets, EDA reports, and ML preparation pipelines.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [AI-First Agent System](#2-ai-first-agent-system)
3. [Project Structure](#3-project-structure)
4. [Setup](#4-setup)
5. [Running the Server](#5-running-the-server)
6. [API Reference](#6-api-reference)
7. [Handler Registry](#7-handler-registry) · [Full Reference →](docs/handler-feature.md)
8. [Data Preparation Pipeline](#8-data-preparation-pipeline)
9. [Code Execution Sandbox](#9-code-execution-sandbox)
10. [Thai Language Support](#10-thai-language-support)
11. [Logging](#11-logging)
12. [Configuration](#12-configuration)
13. [Dependencies](#13-dependencies)

---

## 1. Architecture

```
Web App (Next.js :3000)
        │
        │  POST /chat           →  DS-Agent (AI planner) or Coding Agent
        │  POST /prepare        →  Data Preparation Pipeline (with PrepConfig)
        │  POST /eda-report     →  Structured EDA with auto-charts
        │  POST /suggest-target →  LLM target column suggester
        │  GET  /models         →  Available LLM models list
        │  GET  /health         →  {"status": "ok"}
        ▼
DS-Agent API (FastAPI :8000)
        │
        ├── api/main.py              Entry point — registers routers, CORS, loads .env
        ├── api/models.py            Pydantic v2 request/response models
        ├── api/llm.py               Multi-provider LLM factory (OpenAI + Anthropic, lru_cache)
        ├── api/context.py           Dataset → rich LLM context string builder
        ├── api/sandbox.py           Python exec() sandbox (stdout + Plotly/matplotlib capture)
        ├── api/logger.py            Centralized structured logger → stderr
        │
        ├── api/agents/
        │   ├── datascience.py       DS-Agent orchestrator: plan → execute → interpret (AI-first)
        │   ├── planner.py           AI planner: user message + dataset → JSON plan of steps
        │   ├── step_executor.py     Step executor: handler resolution or codegen fallback
        │   ├── code_generator.py    LLM-powered Python code generation for custom steps
        │   ├── result_interpreter.py LLM result interpreter (codegen results only)
        │   ├── context_analyzer.py  DataContext builder (nulls, skew, cardinality, warnings)
        │   └── coding.py            Coding agent: concise Q&A (no dataset, max_tokens=1024)
        │
        ├── api/handlers/
        │   ├── __init__.py          101-entry HANDLER_REGISTRY
        │   ├── base.py              HandlerResult dataclass, BaseHandler utilities, Thai keywords
        │   ├── stats_handler.py     18 stats functions (describe, correlation, normality_test, etc.)
        │   ├── clean_handler.py     20 cleaning functions (fill nulls, clip outliers, map values, etc.)
        │   ├── transform_handler.py 26 transform functions (filter, pivot, melt, rolling, etc.)
        │   ├── viz_handler.py       22 Plotly chart types (bar, scatter, qq_plot, density, etc.)
        │   └── feature_handler.py   15 feature engineering functions (PCA, mutual_info, etc.)
        │
        ├── api/routes/
        │   ├── chat.py              POST /chat — routes to DS-Agent or Coding Agent
        │   ├── prepare.py           POST /prepare — 10-step prep pipeline with PrepConfig
        │   ├── eda_report.py        POST /eda-report — structured EDA + auto-charts
        │   ├── suggest_target.py    POST /suggest-target
        │   ├── models.py            GET /models — available model list with provider detection
        │   └── handlers_debug.py    GET /debug/handlers (dev only, LOG_LEVEL=debug)
        │
        └── api/data_preparation_agent.py  Full ML data prep pipeline
```

### `/chat` request flow

```
POST /chat
 │
 ├── datasets attached?
 │     YES → run_datascience_agent()
 │            │
 │            ├── Translate Thai keywords → English
 │            ├── Analyze dataset context (nulls, skew, cardinality, warnings)
 │            ├── Check for greeting (trivial shortcut, no AI needed)
 │            │
 │            ├── AI PLANNER (sole decision-maker)
 │            │     ↓ LLM sees full handler catalog (101 handlers with IDs + params)
 │            │     ↓ Outputs JSON plan: each step has handler:{id, params} OR codegen:{task}
 │            │     ↓ No hardcoded keywords, no regex — AI decides everything
 │            │
 │            ├── STEP EXECUTOR (for each step in the plan):
 │            │     ├── handler.id specified → resolve from HANDLER_REGISTRY → execute
 │            │     │     └── Handler fails → silent codegen fallback
 │            │     └── codegen specified → LLM generates Python → sandbox exec
 │            │           └── Auto-retry on error (send error back to LLM once)
 │            │
 │            ├── RESPONSE BUILDER
 │            │     ├── All steps used handlers → return handler summaries directly (no LLM)
 │            │     └── Codegen involved → LLM interpreter explains results with real data
 │            │
 │            └── Smart output routing: query vs generate classification
 │
 └── NO datasets → run_coding_agent()
                   └── LLM general Q&A (temperature 0.3)
```

---

## 2. AI-First Agent System

The DS-Agent uses a fully AI-driven architecture where the **LLM planner is the sole decision-maker**. No hardcoded keywords, no regex patterns, no intent classifiers.

### AI Planner

The planner receives the user message, dataset context, and a **complete handler catalog** (all 101 handlers with their IDs, descriptions, and parameters). It outputs a structured JSON plan where each step specifies either:

- **`handler`** — `{id: "category.sub", params: {...}}` → instant execution via pre-built handler
- **`codegen`** — `{task: "description", produces: "dataframe|chart|text"}` → LLM generates Python code

The planner decides output_type (`query` vs `generate`) and handles disambiguation (e.g., "show nulls" → `stats.null_report` vs "fill nulls" → `clean.fill_nulls`).

### Step Executor

Follows the planner's decisions exactly:
1. **Handler route** — resolves `handler.id` from registry, executes with smart column matching (fuzzy match for user-specified column names)
2. **Codegen route** — LLM generates Python, executed in sandbox with auto-retry on failure
3. **Silent fallback** — if a handler fails or isn't found, automatically falls back to codegen (user never sees errors)

### Response Builder

- **Handler-only results** → uses handler summaries directly (no extra LLM call, fastest path)
- **Codegen or mixed results** → LLM interpreter generates a human-readable explanation using actual computed data (never placeholders)

### Smart Output Routing

Every response is classified by the planner as:
- **`query`** — show inline in chat only, never save as dataset (stats, charts, questions)
- **`generate`** — save as new dataset, add tab to DatasetPicker (cleaning, transforms, data generation)

---

## 3. Project Structure

```
ml-datascience/
├── api/
│   ├── .env                          ← your secrets (git-ignored)
│   ├── .env.example                  ← template
│   ├── main.py                       ← FastAPI app entry point
│   ├── models.py                     ← Pydantic v2 models
│   ├── llm.py                        ← Multi-provider LLM factory (OpenAI + Anthropic)
│   ├── context.py                    ← Dataset context builder
│   ├── sandbox.py                    ← Code execution sandbox (Plotly + matplotlib)
│   ├── logger.py                     ← Centralized logger
│   ├── data_preparation_agent.py     ← ML data prep pipeline
│   │
│   ├── agents/
│   │   ├── datascience.py            ← DS-Agent orchestrator (AI-first, no keyword routing)
│   │   ├── planner.py                ← AI planner: handler catalog + JSON plan output
│   │   ├── step_executor.py          ← Step executor: handler resolution + codegen fallback
│   │   ├── code_generator.py         ← LLM code generation with templates
│   │   ├── result_interpreter.py     ← LLM result interpreter (codegen results only)
│   │   ├── context_analyzer.py       ← DataContext analysis (nulls, skew, cardinality)
│   │   └── coding.py                 ← Coding Q&A agent
│   │
│   ├── handlers/
│   │   ├── __init__.py               ← HANDLER_REGISTRY (55 entries)
│   │   ├── base.py                   ← HandlerResult, BaseHandler, Thai keywords
│   │   ├── stats_handler.py          ← 18 stats functions
│   │   ├── clean_handler.py          ← 20 cleaning functions
│   │   ├── transform_handler.py      ← 26 transform functions
│   │   ├── viz_handler.py            ← 22 Plotly chart types
│   │   └── feature_handler.py        ← 15 feature engineering functions
│   │
│   └── routes/
│       ├── chat.py                   ← POST /chat
│       ├── prepare.py                ← POST /prepare
│       ├── eda_report.py             ← POST /eda-report
│       ├── suggest_target.py         ← POST /suggest-target
│       ├── models.py                 ← GET /models
│       └── handlers_debug.py         ← GET /debug/handlers (dev only)
│
├── requirements.txt
├── Dockerfile
└── start.sh                          ← foreground server launcher
```

---

## 4. Setup

### Prerequisites

- Python 3.10+
- Anthropic API key (required) and/or OpenAI API key (optional)

### Install

```bash
cd ml-datascience
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp api/.env.example api/.env
```

Edit `api/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...        # required
ANTHROPIC_MODEL=claude-sonnet-4-6   # optional — default is claude-sonnet-4-6
OPENAI_API_KEY=sk-proj-...          # optional — enables GPT models in model switcher
LOG_LEVEL=info                      # optional — debug | info | warning | error
```

---

## 5. Running the Server

### Option A — `start.sh` (recommended)

```bash
bash start.sh
```

Loads `api/.env`, activates `.venv`, and starts uvicorn with `--reload`. Press **Ctrl-C** to stop.

### Option B — direct uvicorn

```bash
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Option C — via project launcher

```bash
bash ../installation-core/start.sh
```

Starts both the ML backend and the web app together.

### Verify

```
GET http://localhost:8000/health     → {"status": "ok"}
GET http://localhost:8000/docs       → Swagger UI
GET http://localhost:8000/models     → available models list
```

---

## 6. API Reference

### `POST /chat`

Routes to the DS-Agent (datasets present) or Coding Agent (no datasets).

**Request:**

```json
{
  "message": "Show top 10 rows by SalePrice",
  "datasets": [
    {
      "name": "housing",
      "data": [{ "Id": 1, "SalePrice": 208500, "MSZoning": "RL" }]
    }
  ],
  "conversation_history": [
    { "role": "user", "content": "What columns does this dataset have?" },
    { "role": "assistant", "content": "The dataset has 81 columns..." }
  ],
  "model_id": "claude-sonnet-4-6"
}
```

**Response:**

```json
{
  "response": "Here are the top 10 properties by sale price...",
  "artifacts": {
    "code": "result = df.nlargest(10, 'SalePrice')",
    "chart_json": "<plotly-json>",
    "inline_table": [{ "Id": 1, "SalePrice": 208500 }],
    "data_wrangled": [{ "Id": 1, "SalePrice": 208500 }],
    "dataset_name": "top_10_by_sale_price",
    "dataset_shape": { "rows": 10, "cols": 81 },
    "output_type": "query",
    "should_activate": false
  },
  "model_used": "claude-sonnet-4-6"
}
```

| Field | Description |
|---|---|
| `model_id` | Optional. `"claude-sonnet-4-6"`, `"gpt-4o-mini"`, etc. Auto-detects provider |
| `output_type` | `"query"` (read-only, inline) or `"generate"` (creates new dataset) |
| `should_activate` | Whether frontend should auto-switch to the new dataset |
| `chart_json` | Plotly JSON string (interactive chart) |

---

### `POST /prepare`

Runs the ML data preparation pipeline on a dataset.

**Request:**

```json
{
  "dataset": { "name": "housing", "data": [...] },
  "target_column": "SalePrice",
  "mode": "full",
  "config": {
    "missing_strategy": "auto",
    "scaling_method": "standard",
    "encoding_method": "auto",
    "outlier_treatment": "iqr",
    "outlier_threshold": 1.5,
    "correlation_threshold": 0.95,
    "test_size": 0.2,
    "drop_threshold": 0.4
  }
}
```

| Config Field | Default | Description |
|---|---|---|
| `missing_strategy` | `"auto"` | `auto` \| `mean` \| `median` \| `mode` \| `drop` |
| `scaling_method` | `"standard"` | `standard` \| `minmax` \| `robust` \| `none` |
| `encoding_method` | `"auto"` | `auto` \| `onehot` \| `label` \| `ordinal` |
| `outlier_treatment` | `"iqr"` | `iqr` \| `zscore` \| `none` |
| `correlation_threshold` | `0.95` | Drop one column from highly correlated pairs |
| `test_size` | `0.2` | Train/test split ratio |
| `mode` | `"full"` | `"full"` — ML prep · `"clean"` — cleaning only |

**Response:** Returns `PrepareResponse` with report, split data, feature names, label mappings, and structured `report_detail`.

---

### `POST /eda-report`

Structured EDA report with auto-generated charts.

**Request:**

```json
{
  "dataset": { "name": "housing", "data": [...] }
}
```

**Response:** `EDAResponse` with row/column counts, memory usage, column profiles (null%, unique, stats, skewness), correlation matrix, and auto-generated charts.

---

### `POST /suggest-target`

LLM-powered target column suggestion.

**Request:**

```json
{
  "columns": ["Id", "MSZoning", "LotArea", "SalePrice"],
  "sample_data": [{ "Id": 1, "MSZoning": "RL", "LotArea": 8450, "SalePrice": 208500 }]
}
```

**Response:**

```json
{
  "target_column": "SalePrice",
  "reason": "SalePrice is a continuous numeric column representing the outcome to predict."
}
```

---

### `GET /models`

Returns available LLM models with provider detection based on configured API keys.

```json
[
  { "id": "claude-sonnet-4-6", "label": "Claude Sonnet 4.6", "provider": "anthropic", "badge": "Smart", "available": true },
  { "id": "gpt-4o-mini", "label": "GPT-4o Mini", "provider": "openai", "badge": "Fast", "available": false }
]
```

---

### `GET /health`

```json
{ "status": "ok" }
```

---

## 7. Handler Registry

101 pre-built handlers across 5 categories. The AI planner sees the full catalog and selects handlers by ID.

For the complete reference with all parameters, see **[docs/handler-feature.md](docs/handler-feature.md)**.

### Stats (18 handlers)
`describe`, `shape`, `null_report`, `dtypes`, `value_counts`, `unique_values`, `correlation`, `top_correlations`, `cross_tab`, `class_balance`, `normality_test`, `skewness`, `kurtosis`, `percentile`, `outlier_report`, `duplicate_report`, `zero_report`, `cardinality_report`

### Clean (20 handlers)
`fill_nulls`, `fill_with_value`, `fill_interpolate`, `drop_nulls`, `remove_duplicates`, `deduplicate_by`, `drop_constant`, `clip_outliers`, `remove_outliers`, `drop_column`, `drop_id_columns`, `rename_column`, `lowercase_columns`, `reset_index`, `fix_dtypes`, `change_dtype`, `replace_values`, `map_values`, `lowercase_values`, `strip_whitespace`

### Transform (26 handlers)
`filter`, `sort`, `nlargest`, `nsmallest`, `head`, `tail`, `sample_rows`, `pivot`, `melt`, `groupby_agg`, `rank`, `cumulative`, `rolling`, `round_values`, `add_column`, `assign_value`, `split_column`, `concat_columns`, `encode_label`, `encode_onehot`, `scale_standard`, `scale_minmax`, `scale_robust`, `bin_column`, `qcut`, `inject_null`

### Visualization (22 handlers)
`bar_chart`, `stacked_bar`, `histogram`, `pie_chart`, `count_plot`, `line_chart`, `area_chart`, `scatter`, `distribution`, `density_plot`, `box_plot`, `violin_plot`, `strip_plot`, `qq_plot`, `heatmap`, `pairplot`, `parallel_coords`, `bubble_chart`, `treemap`, `sunburst`, `missing_heatmap`, `time_series`

All charts use a unified minimal Plotly theme — `plotly_white`, `#FB8C3C` accent, Inter font, transparent background.

### Feature Engineering (15 handlers)
`feature_importance`, `mutual_info`, `select_k_best`, `pca`, `correlation_filter`, `variance_filter`, `log_transform`, `sqrt_transform`, `power_transform`, `target_encode`, `frequency_encode`, `cyclical_encode`, `datetime_features`, `ratio_features`, `polynomial_features`

---

## 8. Data Preparation Pipeline

`api/data_preparation_agent.py` runs an LLM-guided ML pipeline:

### Pipeline Steps (full mode)

1. **Drop unusable columns** — all-null, all-constant, IDs (unique ratio > 95%), free-text
2. **Fill missing values** — numeric: median (skewed) or mean (normal); categorical: mode; extreme nulls (> drop_threshold): column dropped
3. **Outlier treatment** — IQR or z-score clipping
4. **Encode categoricals** — label encoding for binary/low-cardinality, one-hot for high-cardinality
5. **Drop correlated features** — pairs exceeding `correlation_threshold`
6. **Scale numeric features** — StandardScaler, MinMaxScaler, or RobustScaler
7. **Train/test split** — stratified for classification, random for regression

Returns structured `PrepReportDetail` with per-step metrics + LLM-generated markdown summary.

---

## 9. Code Execution Sandbox

Code runs inside Python `exec()` with a controlled namespace:

```python
{
    "df":  df,           # primary dataset as pandas DataFrame
    "pd":  pd,           # pandas
    "np":  np,           # numpy
    "plt": plt,          # matplotlib.pyplot (patched to capture figures)
    "sns": sns,          # seaborn
    "px":  px,           # plotly.express
    "go":  go,           # plotly.graph_objects
    "msno": msno,        # missingno
    "make_subplots": ..., # plotly.subplots
    "ff":  ff,           # plotly.figure_factory
}
```

**Key features:**
- **Plotly-first charts** — Plotly figures captured as JSON, with PrepPilot theme applied
- **Thai font support** — auto-detects Tahoma/Arial Unicode MS for matplotlib
- **PrepPilot theme** — orange accent (`#FB8C3C`), Inter/Noto Sans Thai font family
- **Stdout capture** — `contextlib.redirect_stdout` captures all `print()` output
- **Chart capture** — `plt.show` monkey-patched; Plotly figures deduplicated by `id()`
- **DataFrame detection** — scans for result DataFrames in priority order (`result`, `df_result`, `output_df`, etc.)
- **Auto-retry** — if code fails, error sent back to LLM for one fix attempt

---

## 10. Thai Language Support

PrepPilot supports Thai language queries throughout the stack:

- **`THAI_COLUMN_KEYWORDS`** — 30+ Thai → English keyword mappings (ห้องนอน → bedroom, ราคา → price, etc.)
- **`translate_thai_keywords()`** — applied before planner receives the message
- **Thai in planner** — the AI planner prompt includes Thai language examples and a rule to translate Thai intent
- **Thai fonts** — matplotlib uses Tahoma/Arial Unicode MS; Plotly uses `"Inter, Noto Sans Thai, Tahoma"`

---

## 11. Logging

All logs written to **stderr** alongside uvicorn's access logs.

```
22:14:01  INFO   ━━ DS-Agent start ━━  datasets=['housing.csv']
22:14:01  INFO     context: (1460, 81), nulls=19 cols, dupes=0
22:14:02  INFO     Plan: Get dataset dimensions — 1 step(s), output_type=query
22:14:02  INFO     Step 1: Get dataset shape
22:14:02  INFO       route=handler id=stats.shape
22:14:02  INFO   ━━ DS-Agent done  output=query  steps=1  elapsed=1.2s ━━
```

| Log Level | What you see |
|---|---|
| `debug` | Everything including full code blocks, handler params |
| `info` | Step timings, plan details, routes used (default) |
| `warning` | Only warnings and errors |
| `error` | Only errors |

---

## 12. Configuration

All configuration via `api/.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | **Yes** | — | Anthropic API key |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Default model for all LLM calls |
| `OPENAI_API_KEY` | No | — | OpenAI API key (enables GPT models) |
| `LOG_LEVEL` | No | `info` | `debug` / `info` / `warning` / `error` |
| `PORT` | No | `8000` | Port for `start.sh` |
| `RELOAD` | No | `true` | Hot-reload in `start.sh` |

### LLM Factory

`api/llm.py` supports multi-provider model selection:

- **Anthropic**: `claude-haiku-4-5`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-sonnet-4-6`, `claude-opus-4-6`
- **OpenAI**: `gpt-4o-mini`, `gpt-4o`

Auto-detects provider from `model_id`. Uses `functools.lru_cache` keyed on `(provider, api_key, model, temperature, max_tokens)`.

---

## 13. Dependencies

Key packages:

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.129.0 | Web framework |
| `uvicorn` | 0.40.0 | ASGI server |
| `pydantic` | 2.12.5 | Request/response validation (v2) |
| `python-dotenv` | 1.2.1 | `.env` file loading |
| `langchain` | 1.2.10 | LLM orchestration |
| `langchain-anthropic` | ≥0.3.0 | ChatAnthropic integration |
| `langchain-openai` | ≥0.3.0 | ChatOpenAI integration |
| `langchain-core` | 1.2.13 | Base message types |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.2 | Numerical computing |
| `plotly` | 6.5.2 | Interactive charts (primary) |
| `matplotlib` | 3.10.8 | Chart fallback (missingno/sklearn) |
| `seaborn` | 0.13.2 | Statistical plots |
| `missingno` | 0.5.2 | Missing data visualization |
| `scikit-learn` | 1.8.0 | ML pipeline (encoding, scaling, splitting) |
| `scipy` | 1.17.0 | Scientific computing |
| `xgboost` | 3.2.0 | Gradient boosting models |

Full list in `requirements.txt`.

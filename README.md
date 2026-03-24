# ml-datascience — PrepPilot FastAPI Backend

FastAPI service that powers the PrepPilot chat interface. Receives messages and dataset payloads from the web app, runs a multi-step Anthropic Claude LLM pipeline, executes generated Python code in a sandboxed environment, and returns text answers, interactive Plotly charts, result datasets, EDA reports, and ML preparation reports.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Project Structure](#2-project-structure)
3. [Setup](#3-setup)
4. [Running the Server](#4-running-the-server)
5. [API Reference](#5-api-reference)
6. [Agent Logic](#6-agent-logic)
7. [Data Preparation Pipeline](#7-data-preparation-pipeline)
8. [Code Execution Sandbox](#8-code-execution-sandbox)
9. [Artifacts](#9-artifacts)
10. [Logging](#10-logging)
11. [Configuration](#11-configuration)
12. [Dependencies](#12-dependencies)

---

## 1. Architecture

```
Web App (Next.js :3000)
        │
        │  POST /chat           →  DS-Agent or Coding Agent
        │  POST /prepare       →  Data Preparation Pipeline (with PrepConfig)
        │  POST /eda-report    →  Structured EDA with auto-charts
        │  POST /suggest-target → LLM target column suggester
        │  GET  /health        →  {"status": "ok"}
        ▼
DS-Agent API (FastAPI :8000)
        │
        ├── api/main.py          Entry point — registers routers, CORS, loads .env
        ├── api/models.py        Pydantic request/response models
        ├── api/llm.py           Cached ChatAnthropic factory (lru_cache, max_tokens)
        ├── api/context.py       Dataset → LLM context string builder
        ├── api/sandbox.py       Python exec() sandbox (stdout + chart capture)
        ├── api/logger.py        Centralized structured logger → stderr
        │
        ├── api/routes/
        │   ├── chat.py          POST /chat (extracts output_type/should_activate)
        │   ├── prepare.py       POST /prepare (accepts PrepConfig)
        │   ├── eda_report.py    POST /eda-report (structured EDA + charts)
        │   └── suggest_target.py POST /suggest-target
        │
        ├── api/agents/
        │   ├── datascience.py   DS-Agent: full-capability (10 tasks, 18+ charts, auto-retry)
        │   └── coding.py        Coding agent: concise Q&A (no dataset, max_tokens=1024)
        │
        └── api/data_preparation_agent.py  Full ML data prep pipeline
```

### `/chat` request flow

```
POST /chat
 │
 ├── datasets attached?
 │     YES → run_datascience_agent()
 │            ├── Build data context (dtypes, head, stats, value counts)
 │            ├── LLM Step 1 — GPT generates answer or Python code
 │            ├── Sandbox — exec() each code block; capture stdout, charts, DataFrames
 │            ├── LLM Step 2 — GPT interprets real execution output
 │            └── Classify intent → assemble artifacts
 │
 └── NO datasets → run_coding_agent()
                   └── GPT general Q&A (temperature 0.3)
```

---

## 2. Project Structure

```
ml-datascience/
├── api/
│   ├── .env                      ← your secrets (git-ignored)
│   ├── .env.example              ← template
│   ├── main.py                   ← FastAPI app entry point
│   ├── models.py                 ← Pydantic models
│   ├── llm.py                    ← LLM factory (cached)
│   ├── context.py                ← dataset context builder
│   ├── sandbox.py                ← code execution sandbox
│   ├── logger.py                 ← centralized logger
│   ├── data_preparation_agent.py ← ML data prep pipeline
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── datascience.py        ← data-science agent
│   │   └── coding.py             ← coding Q&A agent
│   └── routes/
│       ├── __init__.py
│       ├── chat.py               ← POST /chat
│       ├── prepare.py            ← POST /prepare
│       └── suggest_target.py     ← POST /suggest-target
├── requirements.txt
├── Dockerfile
└── start.sh                      ← foreground server launcher
```

---

## 3. Setup

### Prerequisites

- Python 3.10+
- Anthropic API key

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
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6   # optional — default is claude-sonnet-4-6
LOG_LEVEL=info                      # optional — debug | info | warning | error
```

---

## 4. Running the Server

### Option A — `start.sh` (recommended, logs visible in terminal)

```bash
bash start.sh
```

This loads `api/.env`, activates `.venv`, and starts uvicorn in the foreground with `--reload`. Press **Ctrl-C** to stop.

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
```

---

## 5. API Reference

### `POST /chat`

Routes to the data-science agent (datasets present) or coding agent (no datasets).

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
    { "role": "user",      "content": "What columns does this dataset have?" },
    { "role": "assistant", "content": "The dataset has 81 columns..." }
  ]
}
```

**Response:**

```json
{
  "response": "Here are the top 10 properties by sale price...",
  "artifacts": {
    "code": "result = df.nlargest(10, 'SalePrice')\nprint(result)",
    "code_output": "   Id  SalePrice ...",
    "chart_image": "<base64-encoded PNG>",
    "inline_table": [{ "Id": 1, "SalePrice": 208500 }],
    "data_wrangled": [{ "Id": 1, "SalePrice": 208500 }],
    "dataset_name": "top_10_by_sale_price",
    "dataset_shape": { "rows": 10, "cols": 81 }
  }
}
```

**Notes:**
- `conversation_history` — up to last 20 messages are injected for context
- Multiple datasets can be passed; secondary datasets are available in the sandbox by their sanitized name (e.g. `sales_data.csv` → `sales_data`)
- `datasets` empty → coding agent (no sandbox, no artifacts)

---

### `POST /prepare`

Runs the ML data preparation pipeline on a dataset.

**Request:**

```json
{
  "dataset": {
    "name": "housing",
    "data": [{ "Id": 1, "SalePrice": 208500, "MSZoning": "RL" }]
  },
  "target_column": "SalePrice",
  "test_size": 0.2,
  "scale": true,
  "correlation_threshold": 0.95,
  "mode": "full"
}
```

| Field | Default | Description |
|---|---|---|
| `target_column` | `null` | Column to predict. If omitted, `/suggest-target` logic is used |
| `test_size` | `0.2` | Fraction of data reserved for the test split |
| `scale` | `true` | StandardScaler on numeric features |
| `correlation_threshold` | `0.95` | Drop one column from pairs with correlation above this value |
| `mode` | `"full"` | `"full"` — full ML prep · `"clean"` / `"cleaning"` — cleaning only |

**Response:**

```json
{
  "success": true,
  "mode": "full",
  "report": "## Data Preparation Report\n...",
  "steps": ["Dropped 3 unusable columns", "Filled 12 missing values", "..."],
  "target_column": "SalePrice",
  "target_type": "regression",
  "feature_names": ["MSZoning", "LotArea", "..."],
  "train_rows": 1168,
  "test_rows": 292,
  "n_features": 74,
  "dropped_columns": ["Id", "Alley"],
  "corr_dropped": ["GarageArea"],
  "encoded_columns": ["MSZoning", "Street"],
  "scaled_columns": ["LotArea", "GrLivArea"],
  "label_mappings": { "MSZoning": { "RL": 0, "RM": 1 } },
  "target_label_map": null,
  "X_train": [{ "MSZoning": 0, "LotArea": -0.21 }],
  "X_test":  [{ "MSZoning": 1, "LotArea":  0.45 }],
  "y_train": [208500, 181500],
  "y_test":  [223500]
}
```

---

### `POST /suggest-target`

Uses LLM to identify the most appropriate ML target column from a list of column names and optional sample data.

**Request:**

```json
{
  "columns": ["Id", "MSZoning", "LotArea", "SalePrice"],
  "sample_data": [
    { "Id": 1, "MSZoning": "RL", "LotArea": 8450, "SalePrice": 208500 }
  ]
}
```

**Response:**

```json
{
  "target_column": "SalePrice",
  "reason": "SalePrice is a continuous numeric column representing the outcome to predict in a regression task."
}
```

Falls back to the last column if the LLM call fails.

---

### `GET /health`

```json
{ "status": "ok" }
```

---

### `GET /docs`

Auto-generated Swagger UI. Explore and test all endpoints interactively.

---

## 6. Agent Logic

### Data-Science Agent (`agents/datascience.py`)

**Timing:** Each step is logged with elapsed time for performance monitoring.

**Step 1 — Data context injection**

Before calling the LLM, `context.py` builds a rich string from the dataset:
- Shape (`rows × columns`), memory usage, duplicate row count
- Per-column info: dtype, null%, unique count
- First 5 rows + last 3 rows as markdown tables
- Descriptive statistics + skewness for all numeric columns
- Top-5 value counts for up to 10 categorical columns

**Step 2 — LLM generates answer / code** (`temperature=0.0`)

ChatAnthropic receives: system prompt with data context + last 20 chat history messages + user message. Dynamic `max_tokens`: 2048 for complex tasks (train, EDA, model), 1024 for simple.

The system prompt covers 10 task categories:
1. Data exploration & profiling
2. Data viewing (head/tail/sample)
3. Statistics & aggregation
4. Data cleaning & wrangling (smart: skewed→median, normal→mean)
5. Outlier detection (IQR/z-score)
6. Feature engineering (one-hot, label encode, log transform, binning)
7. Correlation & feature selection
8. Model training (multi-model comparison)
9. Statistical testing (t-test, chi-square, ANOVA)
10. Time series analysis

**Step 3 — Sandbox execution**

All `python` fenced code blocks are extracted and executed (see §8).

**Step 4 — LLM interprets results** (`temperature=0.0`)

A second LLM call (`max_tokens=512`) receives the question, code, stdout output, dataset metadata, and conversation history. Returns a concise answer. If the first code execution fails, an auto-retry sends the error back to the LLM for one fix attempt.

**Step 5 — Intent classification and artifact assembly**

| Intent Keywords | Artifact produced |
|---|---|
| `plot`, `chart`, `graph`, `histogram`, `scatter`, `bar`, `line`, `pie`, `heatmap`, `visualize`, ... | `chart_json` (Plotly) or `chart_image` (PNG fallback) |
| `generate`, `create`, `add`, `modify`, `transform`, `filter`, `clean`, `encode`, `normalize`, ... | `data_wrangled` + `dataset_name` + `dataset_shape` |
| `show`, `display`, `view`, `head`, `tail`, `first`, `last`, `preview`, `sample`, ... | `inline_table` |
| `how many`, `count`, `average`, `mean`, `sum`, `top`, `breakdown`, ... | `inline_table` |

### Coding Agent (`agents/coding.py`)

No dataset. Single ChatAnthropic call (`temperature=0.3`, `max_tokens=1024`). Answers general coding and data science questions concisely with code examples.

---

## 7. Data Preparation Pipeline

`api/data_preparation_agent.py` runs an LLM-guided ML pipeline with three modes:

### Modes

| Mode | What it does |
|---|---|
| `"full"` | Full ML pipeline: drop unusable columns → fill missing → encode categoricals → scale numerics → train/test split |
| `"clean"` / `"cleaning"` | Drop unusable columns + fill missing values only (no encoding, no split) |

### Pipeline steps (full mode)

1. **Drop unusable columns** — removes columns that are all-null, all-constant, IDs (unique ratio > 95%), or free-text (object with avg length > 50 chars)
2. **Fill missing values** — numeric: median; categorical: mode; extreme nulls (> 50%): column dropped
3. **Encode categoricals** — label encoding for binary/low-cardinality, one-hot for high-cardinality (up to 15 categories)
4. **Drop correlated features** — removes one column from each pair exceeding `correlation_threshold` (default 0.95)
5. **Scale numeric features** — `StandardScaler` if `scale=True`
6. **Train/test split** — stratified for classification, random for regression

### LLM-generated report

After pipeline execution, GPT generates a Markdown report summarizing:
- What was cleaned and why
- Encoding and scaling decisions
- Feature engineering choices
- Potential data quality issues to watch for

---

## 8. Code Execution Sandbox

Code runs inside Python `exec()` with a controlled namespace:

```python
{
    "df":  df,    # primary dataset as pandas DataFrame
    "pd":  pd,    # pandas
    "np":  np,    # numpy (imported once at module level)
    "plt": plt,   # matplotlib.pyplot (plt.show patched to capture figures)
    # ... secondary datasets by sanitized variable name
}
```

**Stdout capture:** `contextlib.redirect_stdout` captures all `print()` output.

**Chart capture:** `plt.show` is monkey-patched before `exec()` to save each figure as a base64 PNG string instead of opening a window. Any figures still open after `exec()` are also captured. All figures are closed on exit.

**Last-expression evaluation:** After `exec()`, the last non-comment line is evaluated with `eval()`. If it returns a DataFrame, Series, or scalar, its string representation is appended to stdout.

**DataFrame detection:** After execution, `sandbox.py` scans the namespace for result DataFrames in this priority order:

```
result, df_result, output_df, df_out, df_new, df_filtered,
filtered_df, transformed_df, df_clean, df_transformed
```

Then falls back to any DataFrame not present in the original input.

**Transposed result rejection:** DataFrames with exactly 2 columns where row count equals the original column count are rejected (these are transposed Series, not proper tabular results).

---

## 9. Artifacts

Artifacts are returned in the `artifacts` field of `ChatResponse`. The web app interprets each key differently:

| Key | Type | Description | Saved to DB |
|---|---|---|---|
| `code` | `string` | All Python code blocks that were executed | Yes |
| `code_output` | `string` | Captured stdout from code execution | Yes |
| `chart_image` | `string` | Base64-encoded PNG chart | Yes |
| `inline_table` | `Record[]` | Row data for inline display (show/stats intent) | No — stripped before save |
| `data_wrangled` | `Record[]` | Row data for a new dataset (generate intent) | Converted to dataset, then stripped |
| `dataset_name` | `string` | LLM-generated `snake_case` name for the new dataset | Stored as dataset metadata |
| `dataset_shape` | `object` | `{ rows: number, cols: number }` | Yes |

---

## 10. Logging

All application logs are written to **stderr** so they appear alongside uvicorn's access logs in the same terminal.

Log format:
```
HH:MM:SS  LEVEL     message
```

Example output during a `/chat` request:
```
22:14:01  INFO      ━━ DS-Agent start ━━  datasets=['housing.csv']
22:14:01  INFO        [1/5] loaded primary 'housing.csv'  shape=(1460, 81)
22:14:01  INFO        [2/5] building data context …
22:14:01  INFO        [3/5] calling LLM (step-1: generate answer/code) …
22:14:03  INFO        [3/5] LLM step-1 done  (2.1s)
22:14:03  INFO        [4/5] executing 1 code block(s) in sandbox …
22:14:03  INFO        sandbox block 1 done (0.04s)  result_df=(10, 81)  chart=False  output='...'
22:14:03  INFO        [5/5] calling LLM (step-2: interpret output) …
22:14:05  INFO        [5/5] LLM step-2 done  (1.8s)
22:14:05  INFO        intent  viz=False  generate=False  show=True  stats=False
22:14:05  INFO      ━━ DS-Agent done  total=4.1s ━━
```

### Log level

Set `LOG_LEVEL` in `api/.env`:

| Value | What you see |
|---|---|
| `debug` | Everything including full code blocks sent to sandbox |
| `info` | Normal — step timings, shapes, intents (default) |
| `warning` | Only warnings and errors |
| `error` | Only errors |

---

## 11. Configuration

All configuration is via `api/.env` (loaded automatically by `main.py` via `python-dotenv`):

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key (get one at platform.openai.com) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model name for all LLM calls |
| `LOG_LEVEL` | No | `info` | Logging verbosity: `debug` / `info` / `warning` / `error` |
| `PORT` | No | `8000` | Port for `start.sh` launcher |
| `RELOAD` | No | `true` | Hot-reload in `start.sh` (`true` / `false`) |

### LLM caching

`api/llm.py` uses `functools.lru_cache` keyed on `(api_key, model_name, temperature)`. The `ChatOpenAI` instance is constructed once and reused across all requests, avoiding repeated object construction overhead.

---

## 12. Dependencies

Key packages:

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.129.0 | Web framework |
| `uvicorn` | 0.40.0 | ASGI server |
| `pydantic` | 2.12.5 | Request / response validation |
| `python-dotenv` | 1.2.1 | `.env` file loading |
| `langchain` | 1.2.10 | LLM orchestration |
| `langchain-openai` | 1.1.9 | ChatOpenAI integration |
| `langchain-core` | 1.2.13 | Base message types |
| `langgraph` | 1.0.8 | Agent graph execution |
| `openai` | 2.21.0 | OpenAI API client |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.2 | Numerical computing |
| `matplotlib` | 3.10.8 | Chart generation in sandbox |
| `scikit-learn` | 1.8.0 | ML pipeline (encoding, scaling, splitting) |
| `scipy` | 1.17.0 | Scientific computing |
| `xgboost` | 3.2.0 | Gradient boosting models |
| `mlflow` | 3.9.0 | ML experiment tracking |
| `streamlit` | 1.54.0 | Demo app UI (apps/ only) |
| `gunicorn` | 23.0.0 | Production WSGI server |

Full list in `requirements.txt`.

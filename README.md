# ml-datascience — AI Data Science Agent Backend

FastAPI service that powers the PrepPilot chat interface. It receives messages and dataset payloads from the web app, runs a two-step LLM pipeline, executes generated Python code in a sandbox, and returns answers, charts, and result datasets.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Setup](#2-setup)
3. [API Reference](#3-api-reference)
4. [Agent Logic](#4-agent-logic)
5. [Code Execution Sandbox](#5-code-execution-sandbox)
6. [Artifacts](#6-artifacts)
7. [Configuration](#7-configuration)
8. [Library — ai_data_science_team](#8-library--ai_data_science_team)
9. [Streamlit Apps](#9-streamlit-apps)
10. [Dependencies](#10-dependencies)

---

## 1. Architecture

```
POST /chat
 │
 ├── datasets attached?
 │     YES → run_datascience_agent()
 │            ├── Build data context (dtypes, head, stats, value counts)
 │            ├── LLM Step 1: ChatOpenAI generates answer / Python code
 │            ├── Code Sandbox: exec() with df, pd, plt in scope
 │            │    ├── Capture stdout (print output)
 │            │    ├── Capture matplotlib figures → base64 PNG
 │            │    └── Detect result DataFrames by variable name
 │            ├── LLM Step 2: ChatOpenAI interprets real execution output
 │            └── Return response + artifacts
 │
 └── no datasets → run_coding_agent()
                   └── ChatOpenAI general Q&A (temperature 0.3)
```

---

## 2. Setup

### Prerequisites

- Python 3.10+
- OpenAI API key

### Install

```bash
cd ml-datascience
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Configure

```bash
cp api/.env.example api/.env
```

Edit `api/.env`:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini   # optional
```

### Run

```bash
uvicorn api.main:app --reload --port 8000
```

Or via the project launcher:
```bash
bash ../installation-core/start.sh
```

---

## 3. API Reference

### `POST /chat`

Main endpoint. Routes to data-science agent or coding agent based on whether datasets are provided.

**Request body:**
```json
{
  "message": "Show me the top 10 rows by sale price",
  "datasets": [
    {
      "name": "housing",
      "data": [{ "Id": 1, "SalePrice": 208500, "MSZoning": "RL" }]
    }
  ],
  "conversation_history": [
    { "role": "user", "content": "What columns are in this dataset?" },
    { "role": "assistant", "content": "The dataset has 81 columns..." }
  ]
}
```

**Response body:**
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

**Routing logic:**
- `datasets` non-empty → `run_datascience_agent()`
- `datasets` empty → `run_coding_agent()`

---

### `GET /health`

```json
{ "status": "ok" }
```

---

### `GET /docs`

Auto-generated Swagger UI for exploring the API.

---

## 4. Agent Logic

### Data-Science Agent (`run_datascience_agent`)

**Step 1 — Data context injection**

Before calling the LLM, a rich context string is built from the dataset:
- Shape (rows × columns)
- Column names and dtypes
- First 10 rows (markdown table)
- Descriptive statistics for numeric columns
- Top-5 value counts for up to 8 categorical columns

**Step 2 — LLM answers or generates code**

`ChatOpenAI` (`temperature=0.0`) receives the context + user message + conversation history.
If computation is required, it generates a Python code block using:
- `df` — the dataset as a pandas DataFrame (pre-loaded)
- `pd` — pandas
- `plt` — matplotlib.pyplot (patched to capture figures)

**Step 3 — Code execution**

All Python code blocks are extracted and executed in a sandbox. Results are captured (see §5).

**Step 4 — LLM interprets results**

A second `ChatOpenAI` call receives the original question, the generated code, and the real execution output. It returns a detailed, well-structured natural language answer with a summary.

**Step 5 — Artifact assembly**

Depending on the user's intent:
- **Show intent** (`show`, `display`, `head`, `tail`, `first`, `last`, `preview`, `sample`, etc.) → `inline_table` artifact (rendered in chat, not saved to DB)
- **Compute/transform intent** → `data_wrangled` artifact (saved as a new dataset in MongoDB)
- **Chart request** → `chart_image` artifact (base64 PNG)

### Coding Agent (`run_coding_agent`)

No dataset. Direct `ChatOpenAI` call (`temperature=0.3`). Answers coding and data science questions with explanation, code examples, and a summary.

---

## 5. Code Execution Sandbox

Code runs inside Python's `exec()` with a controlled namespace:

```python
local_ns = {
    "df": df,      # User's dataset as pandas DataFrame
    "pd": pd,      # pandas
    "plt": plt,    # matplotlib.pyplot (patched)
}
```

**Stdout capture:** `contextlib.redirect_stdout` captures all `print()` output.

**Chart capture:** `plt.show` is monkey-patched to save figures as base64 PNG instead of displaying them. Any figures still open after exec are also captured.

**DataFrame detection:** After execution, result DataFrames are found by checking these variable names in priority order:
```
result, df_result, output_df, df_out, df_new, df_filtered,
filtered_df, transformed_df, df_clean, df_transformed
```
Then falls back to scanning all variables for any non-original DataFrame.

**Transposed result rejection:** DataFrames with exactly 2 columns where row count equals the original column count are rejected (these are transposed Series, not proper row data).

**Important rules enforced via system prompt:**
- Always assign DataFrame results to `result = df.xxx` before printing
- Never use `.T`, `.transpose()`, or `.iloc[0]` for show commands
- Assign a distinct color per bar/point/slice in charts

---

## 6. Artifacts

| Key | Type | Description | Persisted in DB? |
|---|---|---|---|
| `code` | `string` | All Python code blocks executed | Yes |
| `code_output` | `string` | Stdout from code execution | Yes |
| `chart_image` | `string` | Base64-encoded PNG chart | Yes |
| `inline_table` | `Record[]` | Rows for inline display (show intent) | No — stripped before save |
| `data_wrangled` | `Record[]` | Rows for new dataset (compute intent) | Converted to dataset, then stripped |
| `dataset_name` | `string` | LLM-generated snake_case name for new dataset | Converted to dataset metadata |
| `dataset_shape` | `object` | `{ rows, cols }` of result dataset | Yes |

---

## 7. Configuration

All configuration is via `api/.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model to use for all LLM calls |

---

## 8. Library — `ai_data_science_team`

The `ai_data_science_team` package (installed via `pip install -e .`) provides reusable LLM-powered agents built on LangChain + LangGraph.

```
ai_data_science_team/
├── agents/
│   ├── data_visualization_agent.py   # Plotly chart generator
│   ├── data_wrangling_agent.py
│   ├── data_cleaning_agent.py
│   └── ...
├── ml_agents/
│   └── model_evaluation_agent.py     # Confusion matrix, metrics
├── ds_agents/
│   └── ...
├── multiagents/
│   └── ...                           # Supervisor / orchestration
├── tools/
│   ├── eda.py                        # EDA utilities
│   └── dataframe.py                  # DataFrame summary helpers
└── utils/
    ├── matplotlib.py                 # base64 chart helpers
    ├── plotly.py                     # Plotly dict utilities
    ├── sandbox.py                    # Subprocess code execution
    └── logging.py                    # Agent logging
```

### Visualization Colors

Charts use the PrepPilot brand palette with one distinct color per bar/slice:

```python
["#FB8C3C", "#4C9BE8", "#2DC88A", "#AB63FA",
 "#FECB52", "#FF6692", "#19D3F3", "#D16C00", "#B6E880", "#F06A6A"]
```

Confusion matrix heatmap: white (`#FFF0E0`) → orange (`#FB8C3C`).

---

## 9. Streamlit Apps

Demo apps in `apps/` — independent of the FastAPI backend:

| App | Path | Description |
|---|---|---|
| AI Pipeline Studio | `apps/ai-pipeline-studio-app/` | Visual multi-agent pipeline builder |
| Exploratory Copilot | `apps/exploratory-copilot-app/` | Interactive EDA with AI guidance |
| Pandas Data Analyst | `apps/pandas-data-analyst-app/` | Conversational pandas agent |
| SQL Database Agent | `apps/sql-database-agent-app/` | SQL query agent |

Run any app:
```bash
streamlit run apps/ai-pipeline-studio-app/app.py
```

---

## 10. Dependencies

Key packages from `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.129.0 | Web framework |
| `uvicorn` | 0.40.0 | ASGI server |
| `pydantic` | 2.12.5 | Request/response validation |
| `langchain` | 1.2.10 | LLM orchestration |
| `langchain-openai` | 1.1.9 | ChatOpenAI integration |
| `langchain-ollama` | 1.0.1 | Local model support |
| `langgraph` | 1.0.8 | Agent graph execution |
| `openai` | 2.21.0 | OpenAI API client |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.2 | Numerical computing |
| `matplotlib` | 3.10.8 | Chart generation |
| `plotly` | 6.5.2 | Interactive charts |
| `scikit-learn` | 1.8.0 | ML utilities |
| `scipy` | 1.17.0 | Scientific computing |
| `h2o` | 3.46.0.9 | AutoML |
| `mlflow` | 3.9.0 | ML experiment tracking |
| `sqlalchemy` | 2.0.46 | SQL database ORM |
| `python-dotenv` | 1.2.1 | `.env` file loading |
| `streamlit` | 1.54.0 | Demo app UI |
| `xgboost` | 3.2.0 | Gradient boosting |
| `gunicorn` | 23.0.0 | Production WSGI server |

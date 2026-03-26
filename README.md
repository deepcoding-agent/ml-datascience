# ml-datascience — PrepPilot FastAPI Backend

FastAPI service powering the PrepPilot AI data science platform. Features an **AI-first agent architecture** with **two-stage routing** — a lightweight category router narrows 393 handlers down to the relevant subset before the planner sees them. No hardcoded keywords or regex routing. **393 pre-built handlers across 7 categories** (stats, clean, transform, viz, feature, NLP, analysis), LLM-powered code generation fallback, and sandboxed Python execution. Supports **Anthropic Claude** and **OpenAI GPT** models with automatic provider detection. Returns text answers, interactive Plotly charts, cleaned datasets, EDA reports, AI insights, comprehensive EDA documents, and ML preparation pipelines.

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
        │  POST /auto-clean    →  AI Auto-Clean (analyze + plan + execute)
        │  POST /auto-prepare  →  AI Auto-Prepare (AI picks config + 10-step pipeline)
        │  POST /prepare        →  Data Preparation Pipeline (with PrepConfig)
        │  POST /eda-report     →  Structured EDA with auto-charts
        │  POST /insights       →  AI-powered dataset insights report
        │  POST /documents      →  Comprehensive EDA document with charts
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
        │   ├── datascience.py       DS-Agent orchestrator: route → plan → execute → interpret
        │   ├── planner.py           Two-stage routing: category router + focused planner
        │   ├── step_executor.py     Step executor: handler resolution or codegen fallback
        │   ├── code_generator.py    LLM-powered Python code generation for custom steps
        │   ├── result_interpreter.py LLM result interpreter (codegen results only)
        │   ├── context_analyzer.py  DataContext builder (nulls, skew, cardinality, warnings)
        │   ├── auto_cleaner.py     AI Auto-Clean: analyze → plan → execute handlers → report
        │   ├── auto_prepare.py     AI Auto-Prepare: analyze → AI picks PrepConfig → pipeline
        │   ├── insights_agent.py   AI Insights: deterministic analysis + LLM narrative
        │   ├── document_agent.py   EDA Document: deep analysis + 6 Plotly charts + AI narrative
        │   └── coding.py            Coding agent: concise Q&A (no dataset, max_tokens=4096)
        │
        ├── api/handlers/
        │   ├── __init__.py          393-entry HANDLER_REGISTRY
        │   ├── base.py              HandlerResult dataclass, BaseHandler utilities
        │   ├── theme.py             Shared Plotly theme (_style, _group_pie, _THEME)
        │   ├── stats/               55 stats handlers (describe, correlation, chi2, CI, Cramer's V, etc.)
        │   ├── clean/               55 cleaning handlers (fill nulls, anonymize, parse JSON, etc.)
        │   ├── transform/           58 transform handlers (filter, pivot, date_diff, row_number, etc.)
        │   ├── viz/                 56 Plotly chart types (bar, scatter, slope, density heatmap, etc.)
        │   ├── feature/             55 feature engineering handlers (PCA, SVD, WoE, percentile, etc.)
        │   ├── nlp/                 57 NLP/text handlers (tokenize, translate, NER, sentiment, etc.)
        │   └── analysis/            57 analysis handlers (clustering, Granger, lift, stationarity, etc.)
        │
        ├── api/routes/
        │   ├── chat.py              POST /chat — routes to DS-Agent or Coding Agent
        │   ├── auto_clean.py        POST /auto-clean — AI-driven dataset cleaning
        │   ├── auto_prepare.py      POST /auto-prepare — AI-driven ML data preparation
        │   ├── prepare.py           POST /prepare — 10-step prep pipeline with PrepConfig
        │   ├── eda_report.py        POST /eda-report — structured EDA + auto-charts
        │   ├── insights.py          POST /insights — AI-powered insights report
        │   ├── documents.py         POST /documents — comprehensive EDA document
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
 │            ├── Analyze dataset context (nulls, skew, cardinality, warnings)
 │            ├── Check for greeting (trivial shortcut, no AI needed)
 │            │
 │            ├── STAGE 1: CATEGORY ROUTER (lightweight, max_tokens=150)
 │            │     ↓ Classifies message into 1-3 handler categories
 │            │     ↓ Categories: stats, clean, transform, viz, feature, nlp, analysis
 │            │     ↓ Or: direct_answer → LLM answers without handlers
 │            │
 │            ├── (direct_answer?) → LLM answers directly with conversation history
 │            │                      (general knowledge, math, coding — not about the dataset)
 │            │
 │            ├── STAGE 2: FOCUSED PLANNER (sees only selected categories)
 │            │     ↓ Receives ~55-170 handlers instead of all 393 (3-7x prompt reduction)
 │            │     ↓ Receives last 6 messages of conversation history (500 chars each)
 │            │     ↓ Outputs JSON plan: each step has handler:{id, params} OR codegen:{task}
 │            │     ↓ Handles Thai natively — no pre-translation needed
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
                   └── LLM general Q&A (temperature 0.3, max_tokens=4096)
```

---

## 2. AI-First Agent System

The DS-Agent uses a fully AI-driven architecture with **two-stage routing**. No hardcoded keywords, no regex patterns, no intent classifiers.

### Two-Stage Routing

With 393 handlers, sending the full catalog to the planner would waste tokens and slow responses. The system uses a two-stage approach:

**Stage 1 -- Category Router** (lightweight, ~150 tokens output):
- A fast LLM call classifies the user message into 1-3 handler categories
- Categories: `stats`, `clean`, `transform`, `viz`, `feature`, `nlp`, `analysis`
- Can also return `direct_answer` for questions not about the dataset
- Input: user message + column names only (no full context)

**Stage 2 -- Focused Planner** (full planning, max_tokens=2048):
- Receives only handlers from the selected categories (~55-170 instead of 393)
- This gives a **3-7x prompt size reduction** compared to the full catalog
- Receives the last 6 messages of conversation history (truncated to 500 chars each) for follow-up context
- Handles Thai language natively -- no pre-translation step needed
- Outputs a structured JSON plan where each step specifies either:
  - **`handler`** -- `{id: "category.sub", params: {...}}` -- instant execution via pre-built handler
  - **`codegen`** -- `{task: "description", produces: "dataframe|chart|text"}` -- LLM generates Python code
- Decides output_type (`query` vs `generate`) and handles disambiguation

### Direct Answer Support

When the router classifies a message as `direct_answer` (general knowledge, math, coding help -- not about the loaded dataset), the system skips the planner entirely and routes to a general-purpose LLM with conversation history. The user gets a natural response without handler overhead.

### Step Executor

Follows the planner's decisions exactly (max_tokens=4096):
1. **Handler route** -- resolves `handler.id` from registry, executes with smart column matching (fuzzy match for user-specified column names)
2. **Codegen route** -- LLM generates Python, executed in sandbox with auto-retry on failure
3. **Silent fallback** -- if a handler fails or isn't found, automatically falls back to codegen (user never sees errors)

### Response Builder

- **Handler-only results** -- uses handler summaries directly (no extra LLM call, fastest path)
- **Codegen or mixed results** -- LLM interpreter (max_tokens=4096) generates a human-readable explanation using actual computed data (never placeholders)

### Smart Output Routing

Every response is classified by the planner as:
- **`query`** -- show inline in chat only, never save as dataset (stats, charts, questions)
- **`generate`** -- save as new dataset, add tab to DatasetPicker (cleaning, transforms, data generation)

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
│   │   ├── datascience.py            ← DS-Agent orchestrator (two-stage routing)
│   │   ├── planner.py                ← Category router + focused planner (393 handlers)
│   │   ├── step_executor.py          ← Step executor: handler resolution + codegen fallback
│   │   ├── code_generator.py         ← LLM code generation with templates
│   │   ├── result_interpreter.py     ← LLM result interpreter (codegen results only)
│   │   ├── context_analyzer.py       ← DataContext analysis (nulls, skew, cardinality)
│   │   ├── auto_cleaner.py          ← AI Auto-Clean agent
│   │   ├── auto_prepare.py          ← AI Auto-Prepare agent
│   │   ├── insights_agent.py        ← AI Insights report (analysis + LLM interpretation)
│   │   ├── document_agent.py        ← EDA Document (deep analysis + 6 charts + AI narrative)
│   │   └── coding.py                 ← Coding Q&A agent (max_tokens=4096)
│   │
│   ├── handlers/                     ← 393 handlers in modular packages
│   │   ├── __init__.py               ← HANDLER_REGISTRY (393 entries)
│   │   ├── base.py                   ← HandlerResult, BaseHandler, Thai keyword map
│   │   ├── theme.py                  ← Shared Plotly theme (_style, _group_pie)
│   │   ├── stats/                    ← 55 stats handlers (one file per handler)
│   │   ├── clean/                    ← 55 cleaning handlers
│   │   ├── transform/                ← 58 transform handlers
│   │   ├── viz/                      ← 56 visualization handlers
│   │   ├── feature/                  ← 55 feature engineering handlers
│   │   ├── nlp/                      ← 57 NLP handlers + _helpers.py
│   │   ├── analysis/                 ← 57 analysis handlers
│   │   ├── code_handler.py           ← Codegen fallback handler
│   │   └── generated/                ← Runtime-generated custom handlers
│   │
│   └── routes/
│       ├── chat.py                   ← POST /chat
│       ├── auto_clean.py             ← POST /auto-clean
│       ├── auto_prepare.py           ← POST /auto-prepare
│       ├── prepare.py                ← POST /prepare
│       ├── eda_report.py             ← POST /eda-report
│       ├── insights.py               ← POST /insights
│       ├── documents.py              ← POST /documents
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
OPENAI_API_KEY=sk-proj-...          # required — default AI provider
OPENAI_MODEL=gpt-4o-mini            # optional — default model
ANTHROPIC_API_KEY=sk-ant-...        # optional — enables Claude models in model switcher
ANTHROPIC_MODEL=claude-sonnet-4-6   # optional — default Claude model
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

### Option C — Docker

```bash
cd installation-core
docker compose up --build
```

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

---

### `POST /insights`

AI-powered dataset insights report with quality score and key findings.

**Request:** `{ "dataset_name": "housing", "data": [...], "model_id": "gpt-4o-mini" }`

**Response:** `{ "success": true, "ai_insights": { "summary": "...", "data_quality_score": 85, "highlights": [...], "recommendations": [...] }, "analysis": {...}, "duration_seconds": 2.3 }`

---

### `POST /documents`

Comprehensive EDA document with AI narrative per section and embedded charts.

**Request:** `{ "dataset_name": "housing", "data": [...], "model_id": "gpt-4o-mini" }`

**Response:** `{ "success": true, "document": { "sections": { "title": "...", "executive_summary": "...", ... }, "charts": { "distribution": "<plotly-json>", "correlation": "...", "missing": "...", ... }, "quality_score": 85, "quality_label": "Excellent" }, "column_profiles": [...], "duration_seconds": 3.1 }`

Charts included: distribution histograms, correlation heatmap, missing values bar, outlier box plots, categorical distributions, dtype pie chart.

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
    "test_size": 0.2
  }
}
```

---

### `POST /auto-clean`

AI-driven dataset cleaning — analyzes, plans, and executes cleaning operations automatically.

### `POST /auto-prepare`

AI-driven ML data preparation — AI selects optimal PrepConfig and runs the full pipeline.

### `POST /eda-report`

Structured EDA report with auto-generated charts.

### `POST /suggest-target`

LLM-powered target column suggestion.

### `GET /models`

Returns available LLM models with provider detection.

### `GET /health`

```json
{ "status": "ok" }
```

---

## 7. Handler Registry

393 pre-built handlers across 7 categories. Each handler is a standalone Python file in its category package. The two-stage router selects relevant categories, then the focused planner picks handlers by ID.

### Stats (55 handlers)
`anova_test`, `cardinality_report`, `chi2_test`, `class_balance`, `cluster_tendency`, `coefficient_variation`, `column_compare`, `confidence_interval`, `correlation`, `correlation_rank`, `covariance`, `cramers_v`, `cross_tab`, `data_sample`, `describe`, `distribution_fit`, `dtypes`, `duplicate_report`, `entropy_report`, `frequency_table`, `gini_report`, `group_stats`, `iqr_report`, `ks_test`, `kurtosis`, `levene_test`, `mann_whitney`, `memory_report`, `missing_pattern`, `mode_report`, `mutual_info_report`, `normality_comprehensive`, `normality_test`, `null_report`, `outlier_report`, `pairwise_stats`, `percentile`, `point_biserial`, `quantile_detail`, `range_report`, `ratio_report`, `shape`, `skewness`, `sparsity_report`, `stability_report`, `summary_extended`, `t_test`, `time_stats`, `top_bottom_values`, `top_correlations`, `unique_values`, `value_counts`, `variance_report`, `z_score_report`, `zero_report`

### Clean (55 handlers)
`anonymize_column`, `cap_outliers_percentile`, `change_dtype`, `clean_column_names`, `clean_currency`, `clean_phone_numbers`, `clean_text_whitespace`, `clip_outliers`, `coalesce_columns`, `dedup_keep_latest`, `deduplicate_by`, `drop_column`, `drop_constant`, `drop_id_columns`, `drop_nulls`, `fill_forward_backward`, `fill_interpolate`, `fill_median_by_group`, `fill_mode`, `fill_nulls`, `fill_with_distribution`, `fill_with_value`, `fix_boolean`, `fix_date_outliers`, `fix_dtypes`, `fix_encoding`, `fix_mixed_types`, `fix_numeric_strings`, `fix_whitespace_names`, `lowercase_columns`, `lowercase_values`, `map_values`, `normalize_text_case`, `parse_json_column`, `remove_duplicates`, `remove_emails`, `remove_empty_rows`, `remove_high_null_cols`, `remove_html_tags`, `remove_negative`, `remove_non_ascii`, `remove_outliers`, `remove_rare_categories`, `remove_special_chars`, `remove_urls`, `remove_zero_rows`, `rename_column`, `replace_values`, `reset_index`, `round_to_nearest`, `split_name`, `standardize_categories`, `standardize_dates`, `strip_whitespace`, `trim_text_length`

### Transform (58 handlers)
`add_column`, `apply_expr`, `assign_value`, `bin_column`, `cast_columns`, `chunk_rows`, `clip`, `concat_columns`, `conditional_column`, `cross_join`, `cumulative`, `date_diff`, `datetime_format`, `datetime_parse`, `drop_rows`, `duplicate_column`, `encode_binary`, `encode_label`, `encode_onehot`, `encode_ordinal`, `explode`, `fill_forward`, `filter`, `flatten_columns`, `groupby_agg`, `head`, `inject_null`, `interpolate_values`, `melt`, `merge`, `nlargest`, `normalize_pct`, `nsmallest`, `pct_change`, `pivot`, `qcut`, `rank`, `reorder_columns`, `resample`, `rolling`, `round_values`, `row_number`, `sample_rows`, `scale_minmax`, `scale_robust`, `scale_standard`, `shift_column`, `shuffle`, `sort`, `split_column`, `stack_columns`, `string_slice`, `tail`, `train_test_split`, `transpose`, `unstack_column`, `where`, `winsorize`

### Visualization (56 handlers)
`area_chart`, `bar_chart`, `box_comparison`, `box_plot`, `bubble_chart`, `candlestick`, `comparison_bar`, `contour_plot`, `correlation_scatter`, `count_plot`, `cumulative_line`, `density_heatmap`, `density_plot`, `distribution`, `donut_chart`, `dot_plot`, `dual_axis`, `ecdf_plot`, `error_bar_chart`, `funnel_chart`, `gauge_chart`, `grouped_bar`, `heatmap`, `histogram`, `histogram_2d`, `icicle_chart`, `line_chart`, `lollipop_chart`, `marimekko`, `missing_heatmap`, `null_bar`, `pairplot`, `parallel_coords`, `pareto_chart`, `percent_bar`, `pie_chart`, `pie_subplots`, `polar_chart`, `qq_plot`, `radar_chart`, `range_plot`, `ridgeline`, `sankey_chart`, `scatter`, `slope_chart`, `stacked_bar`, `step_chart`, `strip_plot`, `sunburst`, `swarm_plot`, `time_series`, `timeline_chart`, `top_n_bar`, `treemap`, `violin_plot`, `waterfall_chart`

### Feature Engineering (55 handlers)
`abs_transform`, `aggregation_features`, `auto_feature_select`, `bin_numeric`, `boxcox_transform`, `clip_features`, `correlation_filter`, `count_encode`, `cumulative_features`, `cyclical_encode`, `datetime_features`, `diff_features`, `distance_from_mean`, `ewm_features`, `exponential_transform`, `feature_cross`, `feature_importance`, `frequency_encode`, `hash_encode`, `interaction_features`, `is_holiday`, `is_null_features`, `is_weekend`, `is_zero_features`, `kbins_discretize`, `label_binarize`, `lag_features`, `log1p_transform`, `log_transform`, `mutual_info`, `ordinal_encode`, `pca`, `percentile_rank`, `polynomial_features`, `power_transform`, `quantile_transform`, `rank_transform`, `rare_category_encode`, `ratio_features`, `reciprocal_transform`, `rolling_stats_features`, `select_k_best`, `sin_cos_hour`, `sqrt_transform`, `string_length_features`, `svd_features`, `target_binary_encode`, `target_encode`, `text_features`, `time_since`, `variance_filter`, `winsorize`, `woe_encode`, `yeo_johnson_transform`, `zscore_features`

### NLP / Text Preprocessing (57 handlers)
`bow`, `char_features`, `class_balance_text`, `collocations`, `doc_term_matrix`, `emoji_features`, `hash_vectorize`, `keyword_extract`, `language_detect`, `named_entity_extract`, `ngrams`, `readability_score`, `regex_extract`, `remove_stopwords`, `sentence_features`, `sentiment_score`, `spelling_features`, `text_augment`, `text_chunk`, `text_clean`, `text_concat`, `text_count_pattern`, `text_dedup`, `text_dedup_exact`, `text_diversity_index`, `text_encode`, `text_extract_numbers`, `text_filter`, `text_label_rules`, `text_length_dist`, `text_lowercase`, `text_mask_pii`, `text_ngram_frequency`, `text_normalize`, `text_oversample`, `text_pattern_match`, `text_pos_patterns`, `text_remove_rare`, `text_replace`, `text_similarity`, `text_split_sentences`, `text_stratified_sample`, `text_summary_report`, `text_title_case`, `text_to_paragraphs`, `text_truncate_pad`, `text_unique_words`, `text_uppercase`, `text_window`, `tfidf`, `tokenize`, `translate`, `vocab_stats`, `word_cloud`, `word_frequency`, `word_overlap`

### Analysis (57 handlers)
`ab_test`, `anomaly_detect`, `auto_eda`, `benchmark_compare`, `bootstrap_ci`, `bottom_n_analysis`, `categorical_analysis`, `categorical_target_crosstab`, `change_point_detect`, `cluster_kmeans`, `cluster_profile`, `cohort_analysis`, `compare_columns`, `compare_extremes`, `concentration_analysis`, `correlation_insights`, `correlation_network`, `cross_correlation`, `data_completeness`, `data_profiling_report`, `data_quality`, `data_readiness_score`, `dbscan_clustering`, `deep_profile`, `diminishing_returns`, `distribution_analysis`, `effect_size`, `feature_drift`, `feature_interaction`, `feature_selection_auto`, `gap_analysis`, `granger_causality`, `group_insights`, `hierarchical_clustering`, `hypothesis_test`, `lift_analysis`, `market_basket`, `missing_value_analysis`, `multicollinearity_check`, `numeric_summary`, `outlier_isolation_forest`, `pareto_analysis`, `pca_2d`, `percentile_analysis`, `prediction_baseline`, `regression_quick`, `rfm_analysis`, `sample_bias_check`, `seasonality_detect`, `segment_analysis`, `sensitivity_analysis`, `stationarity_test`, `survival_curve`, `target_analysis`, `time_series_decompose`, `top_n_analysis`, `trend_detect`, `variance_analysis`

All charts use a unified minimal Plotly theme -- `plotly_white`, `#FB8C3C` accent, Inter font, transparent background.

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

PrepPilot supports Thai language queries natively:

- **No pre-translation step** -- the AI planner and router handle Thai input natively
- **Thai in router** -- the category router prompt includes Thai conjunction detection
- **Thai in planner** -- the planner prompt includes Thai language examples and handles Thai intent directly
- **Thai fonts** -- matplotlib uses Tahoma/Arial Unicode MS; Plotly uses `"Inter, Noto Sans Thai, Tahoma"`

---

## 11. Logging

All logs written to **stderr** alongside uvicorn's access logs.

```
22:14:01  INFO   ━━ DS-Agent start ━━  datasets=['housing.csv']
22:14:01  INFO     context: (1460, 81), nulls=19 cols, dupes=0
22:14:01  INFO     router → categories=['stats'], direct_answer=False
22:14:01  INFO     focused catalog: 1 categories, ~55 handler rows
22:14:02  INFO     Plan: Get dataset dimensions — 1 step(s), output_type=query
22:14:02  INFO     Step 1: Get dataset shape
22:14:02  INFO       route=handler id=stats.shape
22:14:02  INFO   ━━ DS-Agent done  output=query  steps=1  elapsed=1.2s ━━
```

---

## 12. Configuration

All configuration via `api/.env`:

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key (enables Claude models) |
| `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` | Default Claude model |
| `OPENAI_API_KEY` | **Yes** | — | OpenAI API key (default provider) |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Default model for all LLM calls |
| `LOG_LEVEL` | No | `info` | `debug` / `info` / `warning` / `error` |

---

## 13. Dependencies

Key packages:

| Package | Version | Purpose |
|---|---|---|
| `fastapi` | 0.129.0 | Web framework |
| `uvicorn` | 0.40.0 | ASGI server |
| `pydantic` | 2.12.5 | Request/response validation (v2) |
| `langchain` | 1.2.10 | LLM orchestration |
| `langchain-anthropic` | ≥0.3.0 | ChatAnthropic integration |
| `langchain-openai` | ≥0.3.0 | ChatOpenAI integration |
| `pandas` | 2.3.3 | Data manipulation |
| `numpy` | 2.4.2 | Numerical computing |
| `plotly` | 6.5.2 | Interactive charts (primary) |
| `scikit-learn` | 1.8.0 | ML pipeline |
| `scipy` | 1.17.0 | Scientific computing |
| `statsmodels` | 0.14.6 | Statistical tests (ADF, Granger, decomposition) |
| `deep-translator` | 1.11.4 | Google Translate for NLP translate handler |
| `xgboost` | 3.2.0 | Gradient boosting models |

Full list in `requirements.txt` (170+ packages).

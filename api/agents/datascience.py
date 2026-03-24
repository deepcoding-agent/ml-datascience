"""
Data-science agent — handles chat messages when one or more datasets are attached.

Flow
----
1. Build a rich data-context string from every attached dataset.
2. Ask the LLM to answer or generate code  (Step 1).
3. Execute every code block in the sandbox  (Step 2).
4. Ask the LLM to interpret the real execution output  (Step 3).
5. Classify the user intent and decide how to surface the result DataFrame:
     • viz intent      → chart only, no new dataset
     • generate intent → auto-save as new dataset (data_wrangled artifact)
     • show / stats    → inline table
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from api.context import data_context, extract_code_blocks, sanitize_var_name
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload
from api.sandbox import run_code

log = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

DS_SYSTEM_STEP1 = """\
You are an expert data scientist and Python developer. The dataset is ALREADY LOADED \
as a pandas DataFrame called `df`. The actual data is shown below — use it directly.

CORE RULES:
- Be CONCISE. Direct answer first, brief explanation (2-3 sentences).
- NEVER say you lack data access — you have it below.
- Use real values from the data context. Do not make up numbers.
- COLUMN MATCHING: Match user keywords to actual column names from the context. \
ALWAYS verify columns exist in df.columns. Never hallucinate column names. \
Example: user says "garage" → check columns → find "GarageCars" → use "GarageCars".
- LANGUAGE: Respond in the same language the user used. Chart titles too.
- Write ONE clean Python code block when computation is needed.
- `df` is pre-loaded. Do NOT re-import or re-load data.
- Always use print() for output.

AVAILABLE LIBRARIES (pre-imported in sandbox):
- pd (pandas), np (numpy)
- px (plotly.express), go (plotly.graph_objects), make_subplots, ff (plotly.figure_factory)
- plt (matplotlib.pyplot), sns (seaborn), msno (missingno)
- sklearn (import any submodule inside code: sklearn.model_selection, sklearn.preprocessing, etc.)
- scipy.stats (import inside code block)

═══ TASK CAPABILITIES ═══

1. DATA EXPLORATION & PROFILING
When user asks: "describe", "overview", "profile", "EDA", "explore", "summarize", "info"
→ Generate comprehensive profile with nulls, duplicates, dtypes, stats.

2. DATA VIEWING
When user asks: "show", "head", "tail", "sample", "first", "last", "view", "list"
→ ALWAYS: result = df.head(N); print(result)
→ Never print(df.head()) directly. Assign to `result` first.
→ Never use .T or .transpose()

3. STATISTICS & AGGREGATION
When user asks: "how many", "count", "mean", "average", "sum", "total", "median", \
"min", "max", "percentage", "top N", "distribution", "group by", "compare"
→ ALWAYS build a DataFrame for result, then print:
  result = df.groupby('col').agg({{...}}).reset_index(); print(result)
→ NEVER print bare scalars or Series.

4. DATA CLEANING & WRANGLING
When user asks: "clean", "fill", "impute", "drop", "remove", "replace", "rename", \
"filter", "merge", "encode", "normalize", "scale", "transform"
→ ALWAYS: result = df.copy() then modify result, NEVER modify df
→ Print summary: print(f"Before: {{df.shape}}, After: {{result.shape}}")
→ Missing value strategies:
  - Numeric + skewed (|skew| > 1): median
  - Numeric + normal: mean
  - Categorical: mode
  - >40% missing: drop column

5. OUTLIER DETECTION & TREATMENT
When user asks: "outlier", "anomaly", "extreme", "IQR", "z-score"
→ Use IQR method by default: clip values outside Q1-1.5*IQR to Q3+1.5*IQR
→ Print per-column outlier count before/after

6. FEATURE ENGINEERING
When user asks: "feature", "encode", "one-hot", "label encode", "bin", "log transform"
→ One-hot: result = pd.get_dummies(df, columns=['col'], drop_first=True)
→ Label encode: use pd.factorize() or sklearn LabelEncoder
→ Log transform: result['col_log'] = np.log1p(result['col'])
→ Binning: result['col_bin'] = pd.cut(result['col'], bins=5, labels=False)

7. CORRELATION & FEATURE SELECTION
When user asks: "correlation", "relationship", "feature importance", "multicollinearity"
→ Correlation matrix with Plotly heatmap
→ Drop highly correlated features (>0.95)

8. MODEL TRAINING
When user asks: "train", "predict", "classify", "regression", "model", "fit", "ML"
→ Auto-detect task: classification (target is object/few unique) vs regression (numeric)
→ Always split: from sklearn.model_selection import train_test_split
→ Train multiple models, compare metrics as DataFrame
→ Print results sorted by best metric

9. STATISTICAL TESTING
When user asks: "test", "hypothesis", "significant", "p-value", "t-test", "chi-square", "ANOVA"
→ Import from scipy.stats inside code block
→ Print test statistic + p-value + interpretation

10. TIME SERIES
When user asks: "trend", "seasonal", "time series", "forecast", "rolling", "lag"
→ Parse dates first: df['date'] = pd.to_datetime(df['date'])
→ Rolling averages, decomposition, lag features

═══ CHART RULES ═══

Always use Plotly. Assign figure to `fig`. Do NOT call fig.show().
Available: px, go, make_subplots, ff. Do NOT import plotly.

Chart type mapping:
- Bar:          fig = px.bar(data, x='col', y='val', title='...')
- Histogram:    fig = px.histogram(df, x='col', title='...', marginal='box')
- Scatter:      fig = px.scatter(df, x='c1', y='c2', color='c3', title='...', hover_data=[...])
- Line:         fig = px.line(df, x='c1', y='c2', title='...')
- Box:          fig = px.box(df, y='col', title='...')  or  px.box(df, x='group', y='val')
- Violin:       fig = px.violin(df, y='col', title='...')
- Heatmap:      fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdYlBu_r', title='...')
- Pie:          Group >5 categories into "Other" first. fig = px.pie(pie_df, names='cat', values='count', hole=0.3, title='...')
- Scatter matrix: fig = px.scatter_matrix(df, dimensions=[...], title='...')
- Treemap:      fig = px.treemap(df, path=['col1','col2'], values='val', title='...')
- Sunburst:     fig = px.sunburst(df, path=['col1','col2'], values='val', title='...')
- Parallel coords: fig = px.parallel_coordinates(df, dimensions=[...], color='target')
- 3D scatter:   fig = px.scatter_3d(df, x='c1', y='c2', z='c3', color='c4', title='...')
- Subplots:     Use make_subplots(rows=r, cols=c) then fig.add_trace(...)
- Feature imp:  fig = px.bar(imp_df.sort_values('imp'), x='imp', y='feat', orientation='h', title='...')
- Confusion matrix: fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Oranges')
- ROC curve:    fig = px.line(roc_df, x='fpr', y='tpr', title=f'ROC (AUC={{auc:.3f}})')
- Missing vals: fig = px.imshow(df.isnull().astype(int), title='Missing Values', color_continuous_scale=['white','#FF6B35'])

PIE CHART: if >5 categories, group smallest into "Other":
  counts = df['col'].value_counts()
  if len(counts) > 5:
      top5 = counts.head(5)
      counts = pd.concat([top5, pd.Series({{'Other': counts.iloc[5:].sum()}})])
  pie_df = counts.reset_index(); pie_df.columns = ['category', 'count']
  fig = px.pie(pie_df, names='category', values='count', hole=0.3, title='...')
  fig.update_traces(textposition='inside', textinfo='label+percent')

Fallback: matplotlib only for missingno (msno) or sklearn ConfusionMatrixDisplay.
If using matplotlib: plt.tight_layout() before plt.show().

═══ ERROR HANDLING IN CODE ═══
- Wrap sklearn imports in try/except — if not installed, tell user
- Always handle empty DataFrames: check if df.empty or len(df) == 0
- For groupby: use dropna=False to include null groups when relevant
- For to_numeric/to_datetime: always use errors='coerce'

{data_context}
"""

DS_SYSTEM_STEP2 = """\
You are a data science assistant. A user asked a question, code was executed against the dataset,
and the output is shown below. Write a focused, well-structured answer in plain English.

Dataset info:
{dataset_info}

Guidelines:
- Lead with the direct answer or key finding in the first sentence.
- Keep explanations to 2-3 sentences. Do not pad or repeat.
- Use bullet points for multiple findings — no more than 5 bullets.
- Skip the Summary section if the answer is already short (under 4 sentences).
- Do NOT restate the question or say "Based on the output..."
- Reference specific column names and values from the dataset when relevant.
- If the user refers to earlier conversation context, use the history provided.
- LANGUAGE: Respond in the same language the user used. Thai question → Thai answer.

User question: {question}
Code executed:
```python
{code}
```
Execution output:
{output}
"""

# ── Intent keyword sets ───────────────────────────────────────────────────────

_VIZ_KEYWORDS = {
    "plot", "chart", "graph", "histogram", "scatter", "bar", "line",
    "pie", "heatmap", "boxplot", "box", "violinplot", "violin",
    "visualize", "visualise", "visualization", "visualisation",
    "distribution", "correlation", "pairplot", "trend",
}
_GENERATE_KEYWORDS = {
    "generate", "create", "make", "build", "produce", "construct",
    "add", "insert", "put", "introduce", "inject",
    "modify", "transform", "change", "update", "replace", "edit",
    "augment", "simulate", "synthesize", "fabricate",
    "merge", "join", "concat", "combine", "split", "sample",
    "encode", "normalize", "scale", "clean", "impute",
    "rename", "reorder", "sort", "filter", "subset",
}
# These words are ambiguous — they appear in both query and generate contexts.
# "how many nulls" = query,  "add null to 15% of data" = generate.
# Resolved by checking DataFrame shape in determine_output_type().
_AMBIGUOUS_KEYWORDS = {
    "null", "missing", "na", "nan", "duplicate", "drop", "remove",
    "delete", "random", "noise", "new", "shuffle", "fill",
}
_SHOW_KEYWORDS = {
    "show", "display", "view", "print", "head", "tail", "first", "last",
    "list", "preview", "sample", "peek", "look", "see", "what", "rows",
}
_STATS_KEYWORDS = {
    "how", "many", "count", "average", "mean", "median", "sum", "total",
    "percentage", "percent", "top", "bottom", "highest", "lowest",
    "between", "range", "above", "below", "over", "under", "most", "least",
    "number", "much", "often", "frequently", "compare", "breakdown",
    "each", "per", "group", "category", "categories",
}


def _classify_intent(words: set[str], has_charts: bool) -> tuple[bool, bool, bool, bool]:
    """
    Return (is_viz, is_generate, is_show, is_stats) intent flags.
    Viz is checked first and gates the others.
    """
    all_generate = _GENERATE_KEYWORDS | _AMBIGUOUS_KEYWORDS
    is_viz      = bool(words & _VIZ_KEYWORDS) or has_charts
    is_generate = bool(words & all_generate) and not is_viz
    is_show     = bool(words & _SHOW_KEYWORDS) and not is_generate and not is_viz
    is_stats    = (
        bool(words & _STATS_KEYWORDS)
        and not is_generate and not is_viz and not is_show
    )
    return is_viz, is_generate, is_show, is_stats


def _determine_output_type(
    message: str,
    is_generate: bool,
    result_df: pd.DataFrame | None,
) -> str:
    """
    Decides whether a result DataFrame is a real generated dataset ("generate")
    or just a query summary ("query").

    A query summary is a small DataFrame (few rows, few columns) that answers
    a question like "how many nulls". A generated dataset is substantial
    (many rows OR many columns).
    """
    if not is_generate or result_df is None:
        return "query"

    msg = message.lower()
    # If message is clearly a question, it's a query regardless of keywords
    question_starters = ("how many", "what is", "what are", "count of", "show me",
                         "describe", "display", "view", "tell me", "which", "where")
    if any(msg.startswith(q) or msg.startswith(q.replace(" ", "")) for q in question_starters):
        # Unless it also has strong generate verbs
        strong_generate = {"add", "create", "generate", "insert", "merge", "encode",
                           "normalize", "scale", "transform", "fill", "replace",
                           "rename", "filter", "clean", "impute", "split", "augment"}
        if not any(v in msg for v in strong_generate):
            return "query"

    # Small DataFrames (≤10 rows AND ≤3 cols) are likely summaries, not generated data
    if len(result_df) <= 10 and len(result_df.columns) <= 3:
        return "query"

    return "generate"


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
    model_id: str | None = None,
) -> tuple[str, dict]:
    import time
    t0 = time.perf_counter()
    primary = datasets[0]

    # ── Load DataFrames ───────────────────────────────────────────────────────
    log.info("━━ DS-Agent start ━━  datasets=%s", [d.name for d in datasets])
    df = pd.DataFrame(primary.data)
    log.info("  [1/5] loaded primary '%s'  shape=%s", primary.name, df.shape)

    extra_dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets[1:]:
        var = sanitize_var_name(ds.name)
        extra_dfs[var] = pd.DataFrame(ds.data)
        log.info("  [1/5] loaded extra  '%s' → var='%s'  shape=%s", ds.name, var, extra_dfs[var].shape)

    # ── Build data context ────────────────────────────────────────────────────
    log.info("  [2/5] building data context …")
    ctx_parts = [data_context(df, primary.name)]
    for ds, (_, edf) in zip(datasets[1:], extra_dfs.items()):
        ctx_parts.append(data_context(edf, ds.name))

    primary_var = sanitize_var_name(primary.name)
    if len(datasets) > 1:
        var_list = "\n".join(
            f"  - `{sanitize_var_name(ds.name)}` — {ds.name} ({len(ds.data):,} rows)"
            for ds in datasets
        )
        multi_note = (
            f"\nAVAILABLE DATASETS IN SCOPE:\n{var_list}\n"
            f"The primary dataset is also available as `df` (alias for `{primary_var}`).\n"
            f"Use these variable names directly in code — do NOT re-load or re-import them.\n"
        )
    else:
        multi_note = ""

    full_context = multi_note + "\n\n---\n\n".join(ctx_parts)
    system_prompt = DS_SYSTEM_STEP1.format(data_context=full_context)

    history_msgs = build_lc_history(history[-20:]) if history else []

    # ── Step 1: LLM generates answer / code ──────────────────────────────────
    log.info("  [3/5] calling LLM (step-1: generate answer/code) …")
    t_llm1 = time.perf_counter()
    _COMPLEX_KEYWORDS = {"train", "model", "predict", "eda", "profile", "explore",
                         "pipeline", "feature engineering", "hypothesis", "test",
                         "compare models", "cross validation", "confusion matrix"}
    is_complex = any(kw in message.lower() for kw in _COMPLEX_KEYWORDS)
    step1_tokens = 2048 if is_complex else 1024
    llm_step1 = get_llm(temperature=0.0, max_tokens=step1_tokens, model_id=model_id)
    msgs = (
        [SystemMessage(content=system_prompt)]
        + history_msgs
        + [HumanMessage(content=message)]
    )
    step1_reply = llm_step1.invoke(msgs).content
    log.info("  [3/5] LLM step-1 done  (%.1fs)", time.perf_counter() - t_llm1)

    # ── Step 2: execute code blocks ───────────────────────────────────────────
    code_blocks  = extract_code_blocks(step1_reply)
    code_outputs: list[str]          = []
    result_dfs:   list[pd.DataFrame] = []
    chart_images: list[str]          = []
    chart_jsons:  list[str]          = []
    sandbox_dfs:  list[pd.DataFrame] = []
    all_code = ""

    if code_blocks:
        log.info("  [4/5] executing %d code block(s) in sandbox …", len(code_blocks))
    else:
        log.info("  [4/5] no code blocks — skipping sandbox execution")

    for i, block in enumerate(code_blocks, 1):
        log.debug("  sandbox block %d:\n%s", i, block[:400])
        t_exec = time.perf_counter()
        stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(block, df, extra_dfs)
        elapsed = time.perf_counter() - t_exec

        code_outputs.append(stdout)
        if result_df  is not None: result_dfs.append(result_df)
        if chart_b64  is not None: chart_images.append(chart_b64)
        if chart_json is not None: chart_jsons.append(chart_json)
        if sandbox_df is not None: sandbox_dfs.append(sandbox_df)

        if stdout.startswith("Code execution error"):
            log.error("  sandbox block %d error (%.2fs): %s", i, elapsed, stdout)
        else:
            log.info(
                "  sandbox block %d done (%.2fs)  result_df=%s  chart=%s  plotly=%s  output='%s'",
                i, elapsed,
                result_df.shape if result_df is not None else None,
                chart_b64 is not None,
                chart_json is not None,
                stdout[:200],
            )
        all_code = (all_code + "\n" + block).strip()

    # ── Auto-retry on sandbox error ────────────────────────────────────────
    if code_outputs and code_outputs[-1].startswith("Code execution error") and code_blocks:
        log.info("  [4.5/5] sandbox error detected — asking LLM to fix …")
        error_msg = code_outputs[-1]
        fix_prompt = (
            f"The code you generated produced this error:\n{error_msg}\n\n"
            f"Fix the code and try again. Write ONE corrected Python code block. "
            f"Remember: df is already loaded, do NOT re-import data."
        )
        retry_msgs = msgs + [
            AIMessage(content=step1_reply),
            HumanMessage(content=fix_prompt),
        ]
        llm_retry = get_llm(temperature=0.0, max_tokens=1024, model_id=model_id)
        retry_reply = llm_retry.invoke(retry_msgs).content
        retry_blocks = extract_code_blocks(retry_reply)

        if retry_blocks:
            block = retry_blocks[0]
            stdout, result_df, chart_b64, sandbox_df, chart_json = run_code(block, df, extra_dfs)
            if not stdout.startswith("Code execution error"):
                code_outputs[-1] = stdout
                if result_df is not None: result_dfs.append(result_df)
                if chart_b64 is not None: chart_images.append(chart_b64)
                if chart_json is not None: chart_jsons.append(chart_json)
                if sandbox_df is not None: sandbox_dfs.append(sandbox_df)
                all_code = block
                log.info("  [4.5/5] retry succeeded")
            else:
                log.error("  [4.5/5] retry also failed: %s", stdout[:200])

    artifacts: dict[str, Any] = {}
    if all_code:
        artifacts["code"] = all_code
    if chart_images:
        artifacts["chart_image"] = chart_images[0]
        log.info("  chart captured  (total: %d)", len(chart_images))
    if chart_jsons:
        artifacts["chart_json"] = chart_jsons[0]
        log.info("  plotly chart captured  (total: %d)", len(chart_jsons))

    # ── Step 3: LLM interprets execution output ───────────────────────────────
    if code_outputs and all_code:
        combined = "\n---\n".join(code_outputs)
        artifacts["code_output"] = combined
        log.info("  [5/5] calling LLM (step-2: interpret output) …")
        t_llm2 = time.perf_counter()

        # Build compact dataset metadata for the interpretation step
        ds_info_parts = []
        for ds in datasets:
            cols = list(pd.DataFrame(ds.data[:1]).columns) if ds.data else []
            ds_info_parts.append(
                f"- {ds.name}: {len(ds.data):,} rows, {len(cols)} cols → {cols[:15]}"
            )
        dataset_info = "\n".join(ds_info_parts) if ds_info_parts else "(no dataset)"

        interp = DS_SYSTEM_STEP2.format(
            question=message, code=all_code, output=combined,
            dataset_info=dataset_info,
        )
        # Include conversation history so the LLM can resolve follow-up references
        interp_msgs = (
            [SystemMessage(content=interp)]
            + history_msgs
            + [HumanMessage(content=f"Original question: {message}\nInterpret the execution output above.")]
        )
        llm_step2 = get_llm(temperature=0.0, max_tokens=512, model_id=model_id)
        final_text = llm_step2.invoke(interp_msgs).content
        log.info("  [5/5] LLM step-2 done  (%.1fs)", time.perf_counter() - t_llm2)
    else:
        log.info("  [5/5] no code executed — using step-1 reply as final answer")
        final_text = step1_reply

    # ── Step 4: surface result DataFrame ─────────────────────────────────────
    words = set(message.lower().split())
    is_viz, is_generate, is_show, is_stats = _classify_intent(words, bool(chart_images) or bool(chart_jsons))
    log.info(
        "  intent  viz=%s  generate=%s  show=%s  stats=%s",
        is_viz, is_generate, is_show, is_stats,
    )

    # Show/stats fallback: try to parse stdout as a DataFrame when no result_df captured
    if (is_show or is_stats) and not result_dfs and code_outputs:
        try:
            import io as _io
            parsed = pd.read_csv(
                _io.StringIO(code_outputs[0].strip()),
                sep=r"\s{2,}", engine="python",
            )
            if len(parsed) > 0 and len(parsed.columns) >= 2:
                result_dfs.append(parsed)
                log.info("  stdout-parse fallback produced df shape=%s", parsed.shape)
        except Exception:
            pass

    # Generate fallback: use sandbox df if LLM mutated df in-place
    if is_generate and not is_viz and not result_dfs and sandbox_dfs:
        candidate = sandbox_dfs[-1]
        if len(candidate) > 0:
            result_dfs.append(candidate)
            log.info("  sandbox-df fallback  shape=%s", candidate.shape)

    # ── Determine output_type ─────────────────────────────────────────────────
    has_charts = bool(chart_images) or bool(chart_jsons)
    result_df = result_dfs[0] if result_dfs else None

    if result_df is None and not has_charts:
        artifacts["output_type"] = "text"
        artifacts["should_activate"] = False
        log.info("━━ DS-Agent done  output_type=text  total=%.1fs ━━", time.perf_counter() - t0)
        return final_text, artifacts

    real_output = _determine_output_type(message, is_generate, result_df)
    rows_data = result_df.to_dict(orient="records") if result_df is not None else None

    if real_output == "generate" and rows_data is not None:
        # Save as new dataset — user asked to create/modify data
        dataset_name = _generate_dataset_name(message, model_id=model_id)
        artifacts["data_wrangled"] = rows_data
        artifacts["dataset_name"] = dataset_name
        artifacts["dataset_shape"] = {"rows": len(result_df), "cols": len(result_df.columns)}
        artifacts["output_type"] = "chart+dataset" if has_charts else "dataset"
        artifacts["should_activate"] = False  # NEVER auto-activate
        log.info("  artifact: new dataset '%s'  shape=%s", dataset_name, result_df.shape)
    elif has_charts:
        # Query with chart — show inline only
        artifacts["output_type"] = "chart"
        artifacts["should_activate"] = False
        if rows_data is not None:
            artifacts["inline_table"] = rows_data
    elif rows_data is not None:
        # Query result — show inline table only, do NOT save as dataset
        artifacts["inline_table"] = rows_data
        artifacts["output_type"] = "table"
        artifacts["should_activate"] = False
        log.info("  artifact: inline_table  rows=%d  cols=%d", len(result_df), len(result_df.columns))
    else:
        artifacts["output_type"] = "text"
        artifacts["should_activate"] = False

    log.info(
        "━━ DS-Agent done  output_type=%s  real_output=%s  total=%.1fs ━━",
        artifacts.get("output_type"), real_output, time.perf_counter() - t0,
    )
    return final_text, artifacts


def _generate_dataset_name(message: str, model_id: str | None = None) -> str:
    """Ask the LLM for a short snake_case name for the generated dataset."""
    try:
        llm_name = get_llm(temperature=0.0, max_tokens=50, model_id=model_id)
        reply = llm_name.invoke([HumanMessage(
            content=(
                f"Generate a short snake_case dataset name (max 5 words, no spaces) "
                f"describing this data based on the user request: \"{message}\". "
                f"Reply with ONLY the name, nothing else."
            )
        )]).content.strip().replace(" ", "_").lower()
        return re.sub(r"[^a-z0-9_]", "", reply)[:60] or "result_dataset"
    except Exception:
        return "result_dataset"

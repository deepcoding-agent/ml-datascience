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
from langchain_core.messages import HumanMessage, SystemMessage

from api.context import data_context, extract_code_blocks, sanitize_var_name
from api.llm import build_lc_history, get_llm
from api.logger import get_logger
from api.models import ChatMessage, DatasetPayload
from api.sandbox import run_code

log = get_logger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

DS_SYSTEM_STEP1 = """\
You are an expert data scientist. The dataset is ALREADY LOADED into a pandas DataFrame called `df`.
You can see the actual data below — use it to answer directly.

IMPORTANT RULES:
- NEVER say you don't have access to the data. You DO have it — it is shown below.
- Answer using the real numbers from the data context provided.
- When a computation is needed, write ONE clean Python code block.
- In code: `df` is already available. Do NOT re-load or re-import the data.
- Always use print() to output your results in code.
- Do NOT explain what you would do — just do it.
- For show/display/view/list row requests: ALWAYS assign the DataFrame slice to a variable named \
`result` first, then print it. Example: `result = df.head(10); print(result)`. \
Never do `print(df.head(10))` directly. NEVER use .T, .transpose(), or .iloc[0] — \
always return a proper rows × columns DataFrame.
- For count/aggregate/stats questions ("how many", "count", "average", "sum", "total", \
"percentage", "top N", "distribution", "mean", "max", "min", "range", "between"): \
ALWAYS build a proper DataFrame and assign it to `result`, then print it. \
NEVER print a bare scalar or Series. Examples:
    * "how many in each category" → `result = df.groupby('col').size().reset_index(name='count'); print(result)`
    * "how many have price > X"   → `result = pd.DataFrame({{'label': ['count'], 'value': [len(df[df['price'] > X])]}}); print(result)`
    * "average price by bedrooms" → `result = df.groupby('bedrooms')['price'].mean().reset_index(); print(result)`
    * "top 5 by price"            → `result = df.nlargest(5, 'price'); print(result)`
  This ensures the numbers always appear as a structured inline table.
- For generate/modify/transform/create requests (replacing values, adding columns, filling NAs, \
filtering, renaming, etc.): ALWAYS start with `result = df.copy()`, then apply ALL modifications \
to `result` — NEVER modify `df` directly. At the end, print a short summary \
(e.g. `print(result.shape)`). Example: `result = df.copy(); result['price'] = 100; print(result.shape)`.
- For visualization requests (plot, chart, graph, histogram, scatter, bar, line, etc.):
    - Use matplotlib.pyplot (already imported as `plt`) to create the chart.
    - Call plt.tight_layout() BEFORE plt.show() — this prevents label/title overlap.
    - Call plt.show() at the end — the system will capture it automatically.
    - Use plt.figure(figsize=(10, 6)) for a good default size (taller for legends).
    - Always add a title (plt.title) and axis labels where applicable.
    - COLORS: NEVER hardcode a fixed-length color list. Always generate colors that match the \
actual number of data points/categories at runtime:
        * Scatter / line (N points): `colors = plt.cm.tab10(np.linspace(0, 1, len(data)))` \
then pass `color=colors`
        * Bar chart (N bars): `colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))` \
then pass `color=colors`
        * Single-color chart: pass a single string like `color="#FB8C3C"` — do NOT pass a list
        * Categorical hue: use `c=pd.factorize(df['col'])[0]` with `cmap='tab10'`
    - PIE CHARTS: NEVER use both labels= and autopct= on the wedges when there are more than \
4 slices — they overlap. Instead:
        * Use `labels=None` and `autopct=None` on the pie() call itself
        * Move all labels to a legend: \
`plt.legend(labels, loc='best', bbox_to_anchor=(1, 0.5), fontsize=9)`
        * Show percentages inside large slices only: \
`autopct=lambda p: f'{{p:.1f}}%' if p >= 5 else ''`
        * Use `pctdistance=0.75` so percentage text sits cleanly inside the wedge
    - BAR / HORIZONTAL BAR CHARTS: rotate x-axis labels when there are more than 5 categories:
        `plt.xticks(rotation=45, ha='right', fontsize=9)`
    - Always call `plt.tight_layout()` as the very last step before `plt.show()`.

{data_context}
"""

DS_SYSTEM_STEP2 = """\
You are a data science assistant. A user asked a question, code was executed against the dataset,
and the output is shown below. Write a focused, well-structured answer in plain English.

Guidelines:
- Cover all key findings — do not omit important results.
- Interpret the numbers, don't just repeat them (explain what they mean).
- Use bullet points for multiple items; use short paragraphs for explanation.
- Avoid padding, repetition, or over-explaining obvious things.
- End with a **Summary** section (1–2 sentences) that captures the core answer.

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
    "augment", "simulate", "synthesize", "fabricate", "random",
    "new", "missing", "na", "nan", "null", "duplicate", "shuffle",
    "merge", "join", "concat", "combine", "split", "sample",
    "encode", "normalize", "scale", "clean", "impute", "drop",
    "rename", "reorder", "sort", "filter", "subset",
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
    is_viz      = bool(words & _VIZ_KEYWORDS) or has_charts
    is_generate = bool(words & _GENERATE_KEYWORDS) and not is_viz
    is_show     = bool(words & _SHOW_KEYWORDS) and not is_generate and not is_viz
    is_stats    = (
        bool(words & _STATS_KEYWORDS)
        and not is_generate and not is_viz and not is_show
    )
    return is_viz, is_generate, is_show, is_stats


# ── Agent ─────────────────────────────────────────────────────────────────────

def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
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

    llm = get_llm(temperature=0.0)
    history_msgs = build_lc_history(history[-6:]) if history else []

    # ── Step 1: LLM generates answer / code ──────────────────────────────────
    log.info("  [3/5] calling LLM (step-1: generate answer/code) …")
    t_llm1 = time.perf_counter()
    msgs = (
        [SystemMessage(content=system_prompt)]
        + history_msgs
        + [HumanMessage(content=message)]
    )
    step1_reply = llm.invoke(msgs).content
    log.info("  [3/5] LLM step-1 done  (%.1fs)", time.perf_counter() - t_llm1)

    # ── Step 2: execute code blocks ───────────────────────────────────────────
    code_blocks  = extract_code_blocks(step1_reply)
    code_outputs: list[str]          = []
    result_dfs:   list[pd.DataFrame] = []
    chart_images: list[str]          = []
    sandbox_dfs:  list[pd.DataFrame] = []
    all_code = ""

    if code_blocks:
        log.info("  [4/5] executing %d code block(s) in sandbox …", len(code_blocks))
    else:
        log.info("  [4/5] no code blocks — skipping sandbox execution")

    for i, block in enumerate(code_blocks, 1):
        log.debug("  sandbox block %d:\n%s", i, block[:400])
        t_exec = time.perf_counter()
        stdout, result_df, chart_b64, sandbox_df = run_code(block, df, extra_dfs)
        elapsed = time.perf_counter() - t_exec

        code_outputs.append(stdout)
        if result_df  is not None: result_dfs.append(result_df)
        if chart_b64  is not None: chart_images.append(chart_b64)
        if sandbox_df is not None: sandbox_dfs.append(sandbox_df)

        if stdout.startswith("Code execution error"):
            log.error("  sandbox block %d error (%.2fs): %s", i, elapsed, stdout)
        else:
            log.info(
                "  sandbox block %d done (%.2fs)  result_df=%s  chart=%s  output='%s'",
                i, elapsed,
                result_df.shape if result_df is not None else None,
                chart_b64 is not None,
                stdout[:200],
            )
        all_code = (all_code + "\n" + block).strip()

    artifacts: dict[str, Any] = {}
    if all_code:
        artifacts["code"] = all_code
    if chart_images:
        artifacts["chart_image"] = chart_images[0]
        log.info("  chart captured  (total: %d)", len(chart_images))

    # ── Step 3: LLM interprets execution output ───────────────────────────────
    if code_outputs and all_code:
        combined = "\n---\n".join(code_outputs)
        artifacts["code_output"] = combined
        log.info("  [5/5] calling LLM (step-2: interpret output) …")
        t_llm2 = time.perf_counter()
        interp = DS_SYSTEM_STEP2.format(
            question=message, code=all_code, output=combined
        )
        final_text = llm.invoke([SystemMessage(content=interp)]).content
        log.info("  [5/5] LLM step-2 done  (%.1fs)", time.perf_counter() - t_llm2)
    else:
        log.info("  [5/5] no code executed — using step-1 reply as final answer")
        final_text = step1_reply

    # ── Step 4: surface result DataFrame ─────────────────────────────────────
    words = set(message.lower().split())
    is_viz, is_generate, is_show, is_stats = _classify_intent(words, bool(chart_images))
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

    log.info("━━ DS-Agent done  total=%.1fs ━━", time.perf_counter() - t0)

    if not result_dfs:
        return final_text, artifacts

    result_df = result_dfs[0]
    rows_data  = result_df.to_dict(orient="records")

    if is_generate:
        dataset_name = _generate_dataset_name(llm, message)
        artifacts["data_wrangled"]  = rows_data
        artifacts["dataset_name"]   = dataset_name
        artifacts["dataset_shape"]  = {"rows": len(result_df), "cols": len(result_df.columns)}
        log.info("  artifact: new dataset '%s'  shape=%s", dataset_name, result_df.shape)
    else:
        artifacts["inline_table"] = rows_data
        log.info("  artifact: inline_table  rows=%d  cols=%d", len(result_df), len(result_df.columns))

    return final_text, artifacts


def _generate_dataset_name(llm: Any, message: str) -> str:
    """Ask the LLM for a short snake_case name for the generated dataset."""
    try:
        reply = llm.invoke([HumanMessage(
            content=(
                f"Generate a short snake_case dataset name (max 5 words, no spaces) "
                f"describing this data based on the user request: \"{message}\". "
                f"Reply with ONLY the name, nothing else."
            )
        )]).content.strip().replace(" ", "_").lower()
        return re.sub(r"[^a-z0-9_]", "", reply)[:60] or "result_dataset"
    except Exception:
        return "result_dataset"

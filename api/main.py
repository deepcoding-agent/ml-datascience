"""
FastAPI service — coding + data-science agents for the web-app chat interface.

Routing
-------
  • datasets attached  → run_datascience_agent
      Builds a real data context (dtypes, head, stats) from the attached dataset,
      lets the LLM answer directly from actual data, and executes any generated
      Python/pandas code in a local sandbox so computed results are accurate.

  • no datasets → run_coding_agent
      General coding / data-science Q&A via ChatOpenAI.

POST /chat  { message, datasets, conversation_history }
            → { response, artifacts: { code?, code_output?, data_wrangled? } }

Run
---
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from typing import Any

import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
    import matplotlib.pyplot as _plt_mod
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

app = FastAPI(title="DS-Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────

class DatasetPayload(BaseModel):
    name: str
    data: list[dict]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    datasets: list[DatasetPayload] = []
    conversation_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    response: str
    artifacts: dict = {}


class SuggestTargetRequest(BaseModel):
    columns: list[str]
    sample_data: list[dict] = []


class SuggestTargetResponse(BaseModel):
    target_column: str
    reason: str


class PrepareRequest(BaseModel):
    dataset: DatasetPayload
    target_column: str | None = None
    test_size: float = 0.2
    scale: bool = True
    correlation_threshold: float = 0.95
    mode: str = "full"   # "full" | "clean"


class PrepareResponse(BaseModel):
    success: bool
    mode: str = "full"
    report: str
    steps: list[str] = []
    target_column: str = ""
    target_type: str = ""
    feature_names: list[str] = []
    train_rows: int = 0
    test_rows: int = 0
    n_features: int = 0
    dropped_columns: list[str] = []
    corr_dropped: list[str] = []
    encoded_columns: list[str] = []
    scaled_columns: list[str] = []
    label_mappings: dict = {}
    target_label_map: dict | None = None
    X_train: list[dict] = []
    X_test: list[dict] = []
    y_train: list = []
    y_test: list = []
    error: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_llm(temperature: float = 0.0) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def _build_lc_history(history: list[ChatMessage]) -> list:
    out = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
    return out


def _data_context(df: pd.DataFrame, name: str) -> str:
    """
    Build a concise but data-rich context string for the LLM.
    Includes column types, head rows, and descriptive statistics so the model
    can answer questions using REAL numbers from the actual dataset.
    """
    parts: list[str] = []
    parts.append(f"Dataset name : {name}")
    parts.append(f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")

    dtype_lines = "\n".join(f"  {c}: {str(t)}" for c, t in df.dtypes.items())
    parts.append(f"Column types:\n{dtype_lines}")

    try:
        head_md = df.head(10).to_markdown(index=False)
    except Exception:
        head_md = df.head(10).to_string(index=False)
    parts.append(f"First 10 rows:\n{head_md}")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        try:
            stats_md = df[num_cols].describe().round(4).to_markdown()
        except Exception:
            stats_md = df[num_cols].describe().round(4).to_string()
        parts.append(f"Numeric statistics:\n{stats_md}")

    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        cat_lines = []
        for c in cat_cols[:8]:
            vc = df[c].value_counts().head(5)
            cat_lines.append(f"  {c}: {vc.to_dict()}")
        parts.append("Categorical value counts (top 5 each):\n" + "\n".join(cat_lines))

    return "\n\n".join(parts)


def _extract_code_blocks(text: str) -> list[str]:
    """Return all ```python ... ``` blocks from the LLM reply."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def _run_code(
    code: str,
    df: pd.DataFrame,
    extra_dfs: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, pd.DataFrame | None, str | None, pd.DataFrame | None]:
    """
    Execute the code block in a local sandbox with `df` and any extra named
    DataFrames in scope.
    Returns (stdout_output, result_dataframe_or_None, chart_image_base64_or_None, sandbox_df_or_None).
    A result DataFrame is any variable in the sandbox that:
      - is a pd.DataFrame
      - is not one of the input DataFrames
      - has at least one row
    Priority order: result, df_result, output, df_out, df_new, df_filtered,
                    then any other DataFrame variable.
    sandbox_df is the state of `df` after code execution (may differ from input if modified in-place).
    """
    buf = io.StringIO()
    captured_charts: list[str] = []
    input_dfs = {id(df)} | {id(v) for v in (extra_dfs or {}).values()}

    # Patch plt.show to capture figures instead of displaying them
    if _MATPLOTLIB_AVAILABLE:
        def _capture_show() -> None:
            fig = _plt_mod.gcf()
            if fig.get_axes():
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=130)
                img_buf.seek(0)
                captured_charts.append(base64.b64encode(img_buf.read()).decode("utf-8"))
                _plt_mod.close(fig)

        _original_show = _plt_mod.show
        _plt_mod.show = _capture_show  # type: ignore[method-assign]
        local_ns: dict[str, Any] = {"df": df, "pd": pd, "plt": _plt_mod, "np": __import__("numpy")}
    else:
        _original_show = None
        local_ns = {"df": df, "pd": pd, "np": __import__("numpy")}

    # Inject all extra datasets by their sanitized name
    if extra_dfs:
        local_ns.update(extra_dfs)

    try:
        with redirect_stdout(buf):
            exec(compile(code, "<agent_code>", "exec"), local_ns)

        # Capture any figures that were created but never shown
        if _MATPLOTLIB_AVAILABLE:
            for _ in _plt_mod.get_fignums():
                _capture_show()
            _plt_mod.close("all")

        output = buf.getvalue().strip()

        # Try to eval the last non-comment line for inline expressions
        lines = [l for l in code.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
        if lines:
            try:
                val = eval(compile(lines[-1], "<last>", "eval"), local_ns)
                if val is not None:
                    if isinstance(val, pd.DataFrame):
                        result_str = val.head(5).to_markdown(index=False)
                    elif isinstance(val, pd.Series):
                        result_str = val.to_string()
                    else:
                        result_str = str(val)
                    output = (output + "\n" + result_str).strip() if output else result_str
            except Exception:
                pass

        output = output or "(code ran successfully, no output)"

        # Detect a result DataFrame produced by the code
        # A valid result must have rows as records and columns as fields (not transposed)
        def _is_valid_result_df(candidate: pd.DataFrame) -> bool:
            if id(candidate) in input_dfs:
                return False
            if len(candidate) == 0:
                return False
            # Reject 2-column DataFrames that look like transposed Series (index + value)
            if len(candidate.columns) == 2 and len(candidate) == len(df.columns):
                return False
            return True

        priority = ["result", "df_result", "output_df", "df_out", "df_new", "df_filtered",
                    "filtered_df", "transformed_df", "df_clean", "df_transformed"]
        result_df: pd.DataFrame | None = None
        for name in priority:
            val = local_ns.get(name)
            if isinstance(val, pd.DataFrame) and _is_valid_result_df(val):
                result_df = val
                break
        if result_df is None:
            for val in local_ns.values():
                if isinstance(val, pd.DataFrame) and _is_valid_result_df(val):
                    result_df = val
                    break

        chart_image = captured_charts[0] if captured_charts else None
        sandbox_df = local_ns.get("df") if isinstance(local_ns.get("df"), pd.DataFrame) else None
        return output, result_df, chart_image, sandbox_df

    except Exception as exc:
        return f"Code execution error: {exc}", None, None, None
    finally:
        if _MATPLOTLIB_AVAILABLE and _original_show is not None:
            _plt_mod.show = _original_show  # type: ignore[method-assign]
            _plt_mod.close("all")


# ── Coding agent (no datasets) ────────────────────────────────────────────────

CODING_SYSTEM = """\
You are an expert coding and data science assistant.
Give answers that are complete but tight — cover everything that matters, skip what doesn't.
Prefer Python. Use markdown fenced code blocks for any code snippets.
End with a short 1–2 sentence summary of the key takeaway.
"""


def run_coding_agent(message: str, history: list[ChatMessage]) -> str:
    llm = _get_llm(temperature=0.3)
    msgs = [SystemMessage(content=CODING_SYSTEM)]
    msgs += _build_lc_history(history)
    msgs.append(HumanMessage(content=message))
    return llm.invoke(msgs).content


# ── Data-science agent (with datasets) ───────────────────────────────────────

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
- For show/display/view/list row requests: ALWAYS assign the DataFrame slice to a variable named `result` first, then print it. Example: `result = df.head(10); print(result)`. Never do `print(df.head(10))` directly. NEVER use .T, .transpose(), or .iloc[0] — always return a proper rows × columns DataFrame.
- For count/aggregate/stats questions ("how many", "count", "average", "sum", "total", "percentage", "top N", "distribution", "mean", "max", "min", "range", "between"): ALWAYS build a proper DataFrame and assign it to `result`, then print it. NEVER print a bare scalar or Series. Examples:
    * "how many in each category" → `result = df.groupby('col').size().reset_index(name='count'); print(result)`
    * "how many have price > X" → `result = pd.DataFrame({{'label': ['count'], 'value': [len(df[df['price'] > X])]}}); print(result)`
    * "average price by bedrooms" → `result = df.groupby('bedrooms')['price'].mean().reset_index(); print(result)`
    * "top 5 by price" → `result = df.nlargest(5, 'price'); print(result)`
  This ensures the numbers always appear as a structured inline table.
- For generate/modify/transform/create requests (replacing values, adding columns, filling NAs, filtering, renaming, etc.): ALWAYS start with `result = df.copy()`, then apply ALL modifications to `result` — NEVER modify `df` directly. At the end, print a short summary (e.g. `print(result.shape)`). Example: `result = df.copy(); result['price'] = 100; print(result.shape)`.
- For visualization requests (plot, chart, graph, histogram, scatter, bar, line, etc.):
    - Use matplotlib.pyplot (already imported as `plt`) to create the chart.
    - Call plt.tight_layout() BEFORE plt.show() — this prevents label/title overlap.
    - Call plt.show() at the end — the system will capture it automatically.
    - Use plt.figure(figsize=(10, 6)) for a good default size (taller for legends).
    - Always add a title (plt.title) and axis labels where applicable.
    - COLORS: NEVER hardcode a fixed-length color list. Always generate colors that match the actual number of data points/categories at runtime:
        * Scatter / line (N points): `colors = plt.cm.tab10(np.linspace(0, 1, len(data)))` then pass `color=colors`
        * Bar chart (N bars): `colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))` then pass `color=colors`
        * Single-color chart: pass a single string like `color="#FB8C3C"` — do NOT pass a list at all
        * Categorical hue: use `c=pd.factorize(df['col'])[0]` with `cmap='tab10'`
    - PIE CHARTS: NEVER use both labels= and autopct= on the wedges when there are more than 4 slices — they overlap. Instead:
        * Use `labels=None` and `autopct=None` on the pie() call itself
        * Move all labels to a legend: `plt.legend(labels, loc='best', bbox_to_anchor=(1, 0.5), fontsize=9)`
        * Show percentages inside large slices only via a custom autopct that returns '' for slices < 5%:
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


def _sanitize_var_name(name: str) -> str:
    """Convert a dataset name to a valid Python identifier."""
    import re as _re
    s = _re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if s and s[0].isdigit():
        s = "df_" + s
    return s or "df"


def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
) -> tuple[str, dict]:
    primary = datasets[0]
    print(f"[DS-Agent] datasets={[d.name for d in datasets]}", flush=True)

    # ── Load all datasets into DataFrames ────────────────────────────────────
    df = pd.DataFrame(primary.data)
    print(f"[DS-Agent] primary='{primary.name}' shape={df.shape}", flush=True)

    # Build named namespace for all extra datasets
    extra_dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets[1:]:
        var_name = _sanitize_var_name(ds.name)
        extra_dfs[var_name] = pd.DataFrame(ds.data)
        print(f"[DS-Agent] extra '{ds.name}' → variable '{var_name}' shape={extra_dfs[var_name].shape}", flush=True)

    # ── Build data context for ALL datasets ──────────────────────────────────
    context_parts = [_data_context(df, primary.name)]
    for ds, (var_name, edf) in zip(datasets[1:], extra_dfs.items()):
        context_parts.append(_data_context(edf, ds.name))

    # Tell the LLM about all available variable names
    primary_var = _sanitize_var_name(primary.name)
    var_list_str = "\n".join(
        f"  - `{_sanitize_var_name(ds.name)}` — {ds.name} ({len(ds.data):,} rows)"
        for ds in datasets
    )
    multi_note = (
        f"\nAVAILABLE DATASETS IN SCOPE:\n{var_list_str}\n"
        f"The primary dataset is also available as `df` (alias for `{primary_var}`).\n"
        f"Use these variable names directly in code — do NOT re-load or re-import them.\n"
        if len(datasets) > 1 else ""
    )

    full_context = multi_note + "\n\n---\n\n".join(context_parts)
    system_prompt = DS_SYSTEM_STEP1.format(data_context=full_context)
    history_msgs = _build_lc_history(history[-6:]) if history else []

    llm = _get_llm(temperature=0.0)

    # ── Step 1: ask LLM to answer / generate code ──────────────────────────
    msgs = [SystemMessage(content=system_prompt)] + history_msgs + [HumanMessage(content=message)]
    step1_reply = llm.invoke(msgs).content

    # ── Step 2: execute all code blocks ────────────────────────────────────
    code_blocks = _extract_code_blocks(step1_reply)
    code_outputs: list[str] = []
    result_dfs: list[pd.DataFrame] = []
    chart_images: list[str] = []
    sandbox_dfs: list[pd.DataFrame] = []
    all_code = ""
    for block in code_blocks:
        out, result_df, chart_img, sandbox_df = _run_code(block, df, extra_dfs)
        code_outputs.append(out)
        if result_df is not None:
            result_dfs.append(result_df)
        if chart_img is not None:
            chart_images.append(chart_img)
        if sandbox_df is not None:
            sandbox_dfs.append(sandbox_df)
        print(f"[DS-Agent] code_output: {out[:300]}", flush=True)
        all_code = (all_code + "\n" + block).strip()

    artifacts: dict[str, Any] = {}
    if all_code:
        artifacts["code"] = all_code
    if chart_images:
        artifacts["chart_image"] = chart_images[0]  # first chart for now

    # ── Step 3: if code ran, ask LLM to interpret the real output ──────────
    if code_outputs and all_code:
        combined_output = "\n---\n".join(code_outputs)
        artifacts["code_output"] = combined_output

        interp_prompt = DS_SYSTEM_STEP2.format(
            question=message,
            code=all_code,
            output=combined_output,
        )
        final_text = llm.invoke([SystemMessage(content=interp_prompt)]).content
    else:
        final_text = step1_reply

    # ── Step 4: if a new DataFrame was produced, show inline or save as dataset ───

    _msg_lower = message.lower()
    _msg_words = set(_msg_lower.split())

    # Detect visualization intent FIRST — overrides everything else (no new dataset)
    _viz_keywords = {
        "plot", "chart", "graph", "histogram", "scatter", "bar", "line",
        "pie", "heatmap", "boxplot", "box", "violinplot", "violin",
        "visualize", "visualise", "visualization", "visualisation",
        "distribution", "correlation", "pairplot", "trend",
    }
    _is_viz_intent = bool(_msg_words & _viz_keywords) or bool(chart_images)

    # Detect "generate/create/modify" intent → save as new dataset
    # Visualization requests are excluded even if they contain overlapping words
    _generate_keywords = {
        "generate", "create", "make", "build", "produce", "construct",
        "add", "insert", "put", "introduce", "inject",
        "modify", "transform", "change", "update", "replace", "edit",
        "augment", "simulate", "synthesize", "fabricate", "random",
        "new", "missing", "na", "nan", "null", "duplicate", "shuffle",
        "merge", "join", "concat", "combine", "split", "sample",
        "encode", "normalize", "scale", "clean", "impute", "drop",
        "rename", "reorder", "sort", "filter", "subset",
    }
    _is_generate_intent = bool(_msg_words & _generate_keywords) and not _is_viz_intent

    # Detect "show/display/view" intent → show as inline table
    _show_keywords = {"show", "display", "view", "print", "head", "tail", "first", "last",
                      "list", "preview", "sample", "peek", "look", "see", "what", "rows"}
    _is_show_intent = bool(_msg_words & _show_keywords) and not _is_generate_intent and not _is_viz_intent

    # Detect stats/aggregate intent → show computed numbers as inline table
    _stats_keywords = {
        "how", "many", "count", "average", "mean", "median", "sum", "total",
        "percentage", "percent", "top", "bottom", "highest", "lowest",
        "between", "range", "above", "below", "over", "under", "most", "least",
        "number", "much", "often", "frequently", "compare", "breakdown",
        "each", "per", "group", "category", "categories",
    }
    _is_stats_intent = (
        bool(_msg_words & _stats_keywords)
        and not _is_generate_intent
        and not _is_viz_intent
        and not _is_show_intent
    )

    # Fallback: if show/stats intent but no result_df was captured, parse stdout into a DataFrame
    if (_is_show_intent or _is_stats_intent) and not result_dfs and code_outputs:
        try:
            import io as _io
            raw_out = code_outputs[0].strip()
            parsed = pd.read_csv(_io.StringIO(raw_out), sep=r"\s{2,}", engine="python")
            if len(parsed) > 0 and len(parsed.columns) >= 3:
                result_dfs.append(parsed)
        except Exception:
            pass

    # Fallback for generate intent: if LLM modified df in-place (no new result df captured),
    # use the sandbox df as the result — but only if it differs from the original df shape or values.
    # Visualization intent is explicitly excluded — plot code leaves df unchanged on purpose.
    if _is_generate_intent and not _is_viz_intent and not result_dfs and sandbox_dfs:
        sandbox_df_candidate = sandbox_dfs[-1]  # last executed block's df state
        if len(sandbox_df_candidate) > 0:
            result_dfs.append(sandbox_df_candidate)
            print(f"[DS-Agent] fallback: using sandbox df shape={sandbox_df_candidate.shape}", flush=True)

    if result_dfs:
        result_df = result_dfs[0]  # primary result
        rows_data = result_df.to_dict(orient="records")

        if _is_generate_intent:
            # Generate a short snake_case name via LLM, then auto-save to tab bar
            try:
                name_reply = llm.invoke([HumanMessage(
                    content=(
                        f"Generate a short snake_case dataset name (max 5 words, no spaces) "
                        f"that describes what this data is, based on the user request: \"{message}\". "
                        f"Reply with ONLY the name, nothing else."
                    )
                )]).content.strip().replace(" ", "_").lower()
                import re as _re
                dataset_name = _re.sub(r"[^a-z0-9_]", "", name_reply)[:60] or "result_dataset"
            except Exception:
                dataset_name = "result_dataset"

            artifacts["data_wrangled"] = rows_data
            artifacts["dataset_name"] = dataset_name
            artifacts["dataset_shape"] = {"rows": len(result_df), "cols": len(result_df.columns)}
            print(f"[DS-Agent] auto-saving dataset='{dataset_name}' shape={result_df.shape}", flush=True)

        elif _is_show_intent:
            # Show inline with manual Save button
            artifacts["inline_table"] = rows_data
            print(f"[DS-Agent] inline table rows={len(result_df)} cols={len(result_df.columns)}", flush=True)

        else:
            # Default: show inline
            artifacts["inline_table"] = rows_data
            print(f"[DS-Agent] inline table (default) rows={len(result_df)} cols={len(result_df.columns)}", flush=True)

    return final_text, artifacts


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    print(f"[/chat] message='{req.message[:80]}' datasets={[d.name for d in req.datasets]}", flush=True)
    try:
        if req.datasets:
            text, artifacts = run_datascience_agent(
                req.message, req.datasets, req.conversation_history
            )
        else:
            text = run_coding_agent(req.message, req.conversation_history)
            artifacts = {}
        return ChatResponse(response=text, artifacts=artifacts)
    except Exception as exc:
        traceback.print_exc()
        return ChatResponse(
            response=f"Agent error: {exc}",
            artifacts={"error": str(exc)},
        )


@app.post("/prepare", response_model=PrepareResponse)
async def prepare(req: PrepareRequest) -> PrepareResponse:
    from api.data_preparation_agent import run_data_preparation
    print(f"[/prepare] dataset='{req.dataset.name}' rows={len(req.dataset.data)} target='{req.target_column}'", flush=True)
    try:
        result = run_data_preparation(
            data=req.dataset.data,
            target_column=req.target_column,
            test_size=req.test_size,
            scale=req.scale,
            correlation_threshold=req.correlation_threshold,
            mode=req.mode,
        )
        return PrepareResponse(**result)
    except Exception as exc:
        traceback.print_exc()
        return PrepareResponse(
            success=False,
            report=f"## Preparation Failed\n\n**Error:** {exc}",
            error=str(exc),
        )


@app.post("/suggest-target", response_model=SuggestTargetResponse)
async def suggest_target(req: SuggestTargetRequest) -> SuggestTargetResponse:
    """Use LLM to pick the most appropriate target column from the dataset."""
    import json as _json

    cols = req.columns
    if not cols:
        return SuggestTargetResponse(target_column="", reason="No columns provided.")

    sample_str = ""
    if req.sample_data:
        try:
            sample_str = f"\n\nSample rows (first 3):\n{_json.dumps(req.sample_data[:3], default=str, indent=2)}"
        except Exception:
            pass

    col_list = "\n".join(f"  - {c}" for c in cols)
    prompt = (
        "You are a senior data scientist. Identify the single best target column (dependent variable) "
        "for a supervised machine learning task from the columns below.\n\n"
        f"Columns:\n{col_list}{sample_str}\n\n"
        "Selection rules (in priority order):\n"
        "1. Column names that clearly indicate the outcome to predict: "
        "target, label, class, y, output, result, outcome, prediction, response, "
        "price, cost, salary, revenue, sales, churn, survived, diagnosis, fraud, "
        "rating, score, grade, risk, default, approved, status, category, species.\n"
        "2. If multiple candidates, prefer columns with low-to-medium cardinality "
        "(good for classification) or continuous numeric (regression).\n"
        "3. Avoid ID columns, timestamps, free-text, and columns that are clearly "
        "input features (age, height, weight when a health outcome is also present).\n"
        "4. If nothing stands out, pick the last column (common ML convention).\n\n"
        "Reply ONLY with valid JSON, no markdown:\n"
        '{"target_column": "<exact column name>", "reason": "<one concise sentence>"}'
    )

    try:
        llm = _get_llm(temperature=0.0)
        resp = llm.invoke(prompt)
        content = resp.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        data = _json.loads(content)
        target = str(data.get("target_column", cols[-1]))
        # Validate the column actually exists (case-insensitive match)
        matched = next((c for c in cols if c == target), None) or \
                  next((c for c in cols if c.lower() == target.lower()), cols[-1])
        return SuggestTargetResponse(target_column=matched, reason=str(data.get("reason", "")))
    except Exception as exc:
        print(f"[/suggest-target] LLM error: {exc}", flush=True)
        return SuggestTargetResponse(target_column=cols[-1], reason="Using last column as default.")


@app.get("/health")
async def health():
    return {"status": "ok"}

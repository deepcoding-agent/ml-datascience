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


def _run_code(code: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame | None, str | None]:
    """
    Execute the code block in a local sandbox with `df` in scope.
    Returns (stdout_output, result_dataframe_or_None, chart_image_base64_or_None).
    A result DataFrame is any variable in the sandbox that:
      - is a pd.DataFrame
      - is not the original `df`
      - has at least one row
    Priority order: result, df_result, output, df_out, df_new, df_filtered,
                    then any other DataFrame variable.
    """
    buf = io.StringIO()
    captured_charts: list[str] = []

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
        local_ns: dict[str, Any] = {"df": df, "pd": pd, "plt": _plt_mod}
    else:
        _original_show = None
        local_ns = {"df": df, "pd": pd}

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
            if candidate is df:
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
        return output, result_df, chart_image

    except Exception as exc:
        return f"Code execution error: {exc}", None, None
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
- For numeric questions ("largest", "max", "average", etc.) use ONLY numeric columns:
    df.select_dtypes(include='number')
- Always use print() to output your results in code.
- Do NOT explain what you would do — just do it.
- For show/display/view/list row requests: ALWAYS assign the DataFrame slice to a variable named `result` first, then print it. Example: `result = df.head(10); print(result)`. Never do `print(df.head(10))` directly. NEVER use .T, .transpose(), or .iloc[0] — always return a proper rows × columns DataFrame.
- For visualization requests (plot, chart, graph, histogram, scatter, bar, line, etc.):
    - Use matplotlib.pyplot (already imported as `plt`) to create the chart.
    - Call plt.show() at the end — the system will capture it automatically.
    - Use plt.figure(figsize=(10, 5)) for a good default size.
    - Always label axes and add a title.
    - Assign a distinct color to EACH individual bar/point/slice so every value is visually distinguishable. Use this palette cycling through as needed: ["#FB8C3C", "#4C9BE8", "#2DC88A", "#AB63FA", "#FECB52", "#FF6692", "#19D3F3", "#D16C00", "#B6E880", "#F06A6A"]. Pass the list to the `color` parameter (e.g. bar colors via a list).

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


def run_datascience_agent(
    message: str,
    datasets: list[DatasetPayload],
    history: list[ChatMessage],
) -> tuple[str, dict]:
    primary = datasets[0]
    print(f"[DS-Agent] dataset='{primary.name}' rows={len(primary.data)}", flush=True)

    df = pd.DataFrame(primary.data)
    print(f"[DS-Agent] df.shape={df.shape} columns={list(df.columns)}", flush=True)

    context = _data_context(df, primary.name)
    system_prompt = DS_SYSTEM_STEP1.format(data_context=context)
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
    all_code = ""
    for block in code_blocks:
        out, result_df, chart_img = _run_code(block, df)
        code_outputs.append(out)
        if result_df is not None:
            result_dfs.append(result_df)
        if chart_img is not None:
            chart_images.append(chart_img)
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

    # Detect "show/display/view" intent
    _show_keywords = {"show", "display", "view", "print", "head", "tail", "first", "last",
                      "list", "preview", "sample", "peek", "look", "see", "give", "what", "rows"}
    _msg_lower = message.lower()
    _msg_words = set(_msg_lower.split())
    _is_show_intent = bool(_msg_words & _show_keywords)

    # Fallback: if show intent but no result_df was captured (e.g. LLM did print(df.head())),
    # try to parse stdout output back into a proper rows×columns DataFrame
    if _is_show_intent and not result_dfs and code_outputs:
        try:
            import io as _io
            raw_out = code_outputs[0].strip()
            parsed = pd.read_csv(_io.StringIO(raw_out), sep=r"\s{2,}", engine="python")
            # Reject transposed/Series output: valid table has many columns (>= original df cols or >= 3)
            # A transposed result has only 2 columns (index + value)
            if len(parsed) > 0 and len(parsed.columns) >= 3:
                result_dfs.append(parsed)
        except Exception:
            pass

    if result_dfs:
        result_df = result_dfs[0]  # primary result
        rows_data = result_df.to_dict(orient="records")

        if _is_show_intent:
            artifacts["inline_table"] = rows_data
            print(f"[DS-Agent] inline table rows={len(result_df)}", flush=True)
        else:
            # Generate a short snake_case name via LLM
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
            print(f"[DS-Agent] result dataset='{dataset_name}' shape={result_df.shape}", flush=True)

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


@app.get("/health")
async def health():
    return {"status": "ok"}

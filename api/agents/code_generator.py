"""Code Generator — generates executable Python code for plan steps.

Called by the step executor when no pre-built handler matches.
Uses LLM to write code that operates on the current DataFrame.
"""
from __future__ import annotations

import pandas as pd

from api.llm import format_history_for_prompt
from api.logger import get_logger

log = get_logger(__name__)

CODE_GEN_PROMPT = """\
Write Python code for this data-science task:
"{step_description}"

## DATAFRAME CONTEXT
{df_context}

## COLUMN DTYPES
{dtypes}

## CONVERSATION HISTORY (recent turns — use only if the task references them)
{conversation_history}

> History rule: most tasks are self-contained — IGNORE the history block above
> unless the current task explicitly leans on prior context (pronouns like
> "it/them/that", references to "the previous result", "same column",
> "drop X too", "now compare it with Y"). If the task is self-contained,
> behave as if no history exists.

## MANDATORY RULES
1. The DataFrame is already loaded as `df` — NEVER reload or redefine it.
2. Use ONLY actual column names from the dtypes list above — NEVER invent columns.
3. Use `df is None` to check for None — NEVER use `if not df` (pandas ambiguity error).
4. Use `format="mixed"` for any `pd.to_datetime()` call.
5. For charts: assign to `fig` using plotly — NEVER call `fig.show()`.
6. For new/modified DataFrames: assign to `result`.
7. For text output: use `print()`.
8. Wrap everything in try/except — print error message on exception.
9. PLOTLY SAFETY: Always convert Interval/Category/mixed types to `str` before
   passing to px.bar/px.pie etc. Use `.astype(str)` or `[str(x) for x in values]`.
10. NEVER import pandas or numpy — they are already available as `pd` and `np`.
11. For plotly: `px` and `go` are already available — just use them directly.

## WHAT THIS STEP PRODUCES
{produces}

## PREVIOUS ERROR (if retrying)
{previous_error}

## TASK-SPECIFIC TEMPLATES

### If injecting null values:
```python
result = df.copy()
fraction = 0.15  # use actual percentage from task
for col in result.columns:
    n = int(len(result) * fraction)
    idx = np.random.choice(result.index, size=n, replace=False)
    result.loc[idx, col] = np.nan
print(f"Injected nulls: {{result.isnull().sum().sum()}} total cells")
print(f"Null percentage: {{result.isnull().mean().mean()*100:.1f}}%")
```

### If binning/splitting into ranges:
```python
col_name = 'ActualColumnName'  # use real column name
bins = pd.cut(df[col_name], bins=N)
counts = bins.value_counts().sort_index()

def _fmt(n):
    if abs(n) >= 1_000_000: return f"{{n/1_000_000:.1f}}M"
    if abs(n) >= 1_000: return f"{{int(n/1_000)}}K"
    return f"{{n:.0f}}"

labels = [f"{{_fmt(iv.left)}} – {{_fmt(iv.right)}}" for iv in counts.index]
pcts = (counts / counts.sum() * 100).round(2)
result = pd.DataFrame({{'Range': labels, 'Count': counts.values, 'Percentage': pcts.values}})
print(result.to_string(index=False))

fig = px.bar(result, x='Range', y='Count', title=f'{{col_name}} Distribution', text='Count')
fig.update_traces(texttemplate='%{{text:,}}', textposition='outside', marker_color='#FB8C3C')
fig.update_layout(template='plotly_white', xaxis=dict(tickangle=0), bargap=0.3)
```

### If creating a chart:
```python
fig = px.chart_type(df, x='actual_col', y='actual_col', title='Descriptive Title')
fig.update_layout(template="plotly_white")
# Title should describe the data, not the internal task
```

### If computing statistics/aggregations:
```python
# Do the computation
result_value = df['col'].mean()
print(f"Mean of col: {{result_value:,.2f}}")
# If producing a table:
result = pd.DataFrame({{...}})
print(result.to_string(index=False))
```

### If comparing rows (max vs min, top vs bottom):
```python
col = 'TargetColumn'
max_row = df.loc[df[col].idxmax()]
min_row = df.loc[df[col].idxmin()]
print(f"Highest {{col}}: {{max_row[col]:,.2f}}")
print(f"Lowest {{col}}: {{min_row[col]:,.2f}}")
for c in df.columns:
    print(f"  {{c}}: {{max_row[c]}} vs {{min_row[c]}}")
# Side-by-side table
result = pd.DataFrame({{'Column': df.columns, 'Highest': [max_row[c] for c in df.columns], 'Lowest': [min_row[c] for c in df.columns]}})
```

### If filtering and summarizing groups:
```python
# Group, aggregate, then format
grouped = df.groupby('group_col')['value_col'].agg(['count','mean','median','min','max']).round(2)
grouped = grouped.sort_values('mean', ascending=False).reset_index()
result = grouped
print(result.to_string(index=False))
fig = px.bar(result, x='group_col', y='mean', text='count', title='Group Comparison')
```

### If doing multi-step analysis (compare, correlate, segment):
```python
# Step 1: compute what's needed
# Step 2: create summary table as result DataFrame
# Step 3: print key findings with actual numbers
# Step 4: create Plotly chart if visual helps
print(f"Key finding 1: ...")
print(f"Key finding 2: ...")
result = pd.DataFrame(summary_data)
fig = px.bar(result, ...)
```

## STYLE GUIDE FOR CODE OUTPUT
- Print key findings with exact numbers — the interpreter uses these to build the response
- Use format specifiers: {{value:,.2f}} for numbers, {{value:.1f}}% for percentages
- Create a `result` DataFrame when the user would benefit from seeing a table
- Create a `fig` Plotly chart ONLY when a visualization would be meaningful — same
  data-suitability gates as the planner: skip charts when the target column is
  freeform text / unique-per-row / constant / has < 3 distinct values, or when the
  dataset is prose/notes/schemas. Returning just `result` and `print(...)` is the
  correct answer for many requests; do NOT force a `fig` to "be helpful". An empty
  axis is worse than no chart.
- Do NOT print raw DataFrames with df.to_string() — print human-readable summaries instead

## OUTPUT
Return ONLY executable Python code. No markdown fences. No explanation.
"""


def generate_step_code(
    step_description: str,
    df_context: str,
    current_df: pd.DataFrame,
    produces: str,
    llm,
    previous_error: str | None = None,
    history: list | None = None,
) -> str:
    """Generate executable Python code for a single step."""
    dtypes_str = "\n".join(
        f"  {col}: {dtype}" for col, dtype in current_df.dtypes.items()
    )

    prompt = CODE_GEN_PROMPT.format(
        step_description=step_description,
        df_context=df_context,
        dtypes=dtypes_str,
        produces=produces,
        previous_error=previous_error or "None — first attempt",
        conversation_history=format_history_for_prompt(history),
    )

    response = llm.invoke(prompt)
    code = response.content.strip()

    # Strip markdown fencing
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    log.info("  codegen: %s (%d chars)", step_description[:60], len(code))
    return code

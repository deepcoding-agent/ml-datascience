"""Code Generator — generates executable Python code for plan steps.

Called by the step executor when no pre-built handler matches.
Uses LLM to write code that operates on the current DataFrame.
"""
from __future__ import annotations

import pandas as pd

from api.logger import get_logger

log = get_logger(__name__)

CODE_GEN_PROMPT = """\
Write Python code for this data-science task:
"{step_description}"

## DATAFRAME CONTEXT
{df_context}

## COLUMN DTYPES
{dtypes}

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

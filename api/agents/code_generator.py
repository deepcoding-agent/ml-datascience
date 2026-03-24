"""Code Generator — generates executable Python code for individual plan steps.

Called by the step executor when no pre-built handler matches.
Uses LLM to write code that operates on the current DataFrame.
"""
from __future__ import annotations

import pandas as pd

from api.logger import get_logger

log = get_logger(__name__)

CODE_GEN_PROMPT = """\
Write Python code to perform this specific step:
"{step_description}"

## CURRENT DATAFRAME STATE
{df_context}

## COLUMN DTYPES
{dtypes}

## RULES — ALL REQUIRED
1. Input DataFrame is available as variable `df` — do NOT reload it
2. Never hardcode column names — use actual columns from context above
3. Use `df is None` check, never `if not df`
4. Use format="mixed" for pd.to_datetime()
5. For charts: assign to `fig` — never call fig.show()
6. For new/modified DataFrame: assign to `result`
7. For text output: use print()
8. Wrap everything in try/except — print error on exception
9. PLOTLY TYPE SAFETY: Always convert Interval/Category/mixed types to str
   before passing to px.bar/px.pie x or names. Use .astype(str) or
   [str(x) for x in values]. Never mix float and str in the same axis.

## WHAT THIS STEP SHOULD PRODUCE
{produces}

## PREVIOUS ERROR (if retrying)
{previous_error}

## SPECIFIC GUIDANCE BY TASK TYPE

If injecting nulls:
    import numpy as np
    df_new = df.copy()
    fraction = 0.15  # or whatever % user specified
    for col in df_new.columns:
        n = int(len(df_new) * fraction)
        idx = np.random.choice(df_new.index, size=n, replace=False)
        df_new.loc[idx, col] = np.nan
    result = df_new
    print(f"Injected nulls: {{result.isnull().sum().sum()}} total null cells")
    print(f"Null percentage: {{result.isnull().mean().mean()*100:.1f}}%")

If binning/splitting:
    bins = pd.cut(df['ColumnName'], bins=N)
    counts = bins.value_counts().sort_index()
    labels = [str(x) for x in counts.index]  # MUST convert to str
    pcts = (counts / counts.sum() * 100).round(2)
    result = pd.DataFrame({{'Range': labels, 'Count': counts.values, 'Percentage': pcts.values}})
    print(result.to_string(index=False))

If generating chart:
    import plotly.express as px
    fig = px.chart_type(df, x='col', y='col', title='...')
    fig.update_layout(template="plotly_white")

## OUTPUT
Return ONLY executable Python code. No markdown, no explanation.
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
        previous_error=previous_error or "None",
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

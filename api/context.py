"""Dataset context utilities: context string builder, name sanitiser, code extractor."""
from __future__ import annotations

import re

import pandas as pd


def data_context(df: pd.DataFrame, name: str) -> str:
    """
    Build a concise but data-rich context string for the LLM.
    Includes shape, column types, first 10 rows, numeric stats,
    and top-5 value counts for categorical columns.
    """
    parts: list[str] = [
        f"Dataset name : {name}",
        f"Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns",
        "Column types:\n" + "\n".join(f"  {c}: {t}" for c, t in df.dtypes.items()),
    ]

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
        cat_lines = [
            f"  {c}: {df[c].value_counts().head(5).to_dict()}"
            for c in cat_cols[:8]
        ]
        parts.append("Categorical value counts (top 5 each):\n" + "\n".join(cat_lines))

    return "\n\n".join(parts)


def sanitize_var_name(name: str) -> str:
    """Convert a dataset name to a valid Python identifier."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    if s and s[0].isdigit():
        s = "df_" + s
    return s or "df"


def extract_code_blocks(text: str) -> list[str]:
    """Return all ```python ... ``` fenced code blocks from an LLM reply."""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)

"""Dataset context utilities: context string builder, name sanitiser, code extractor."""
from __future__ import annotations

import re

import pandas as pd


def data_context(df: pd.DataFrame, name: str) -> str:
    """
    Build a rich context string for the LLM.
    Includes shape, memory, duplicates, per-column info (dtype, null%, unique),
    first 5 + last 3 rows, numeric stats with skewness, and categorical top values.
    """
    parts: list[str] = [
        f"Dataset: {name}",
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        f"Duplicate rows: {df.duplicated().sum():,}",
    ]

    # Column info: name, dtype, nulls, unique
    col_info = []
    for c in df.columns:
        null_pct = df[c].isnull().mean() * 100
        uniq = df[c].nunique()
        col_info.append(f"  {c}: {df[c].dtype} | nulls={null_pct:.1f}% | unique={uniq}")
    parts.append("Columns:\n" + "\n".join(col_info))

    # First 5 + last 3 rows
    try:
        parts.append(f"First 5 rows:\n{df.head(5).to_markdown(index=False)}")
        parts.append(f"Last 3 rows:\n{df.tail(3).to_markdown(index=False)}")
    except Exception:
        parts.append(f"First 5 rows:\n{df.head(5).to_string(index=False)}")

    # Numeric stats + skewness
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(2)
        try:
            skew_row = df[num_cols].skew().round(2).to_frame("skewness").T
            stats = pd.concat([desc, skew_row])
        except Exception:
            stats = desc
        try:
            parts.append(f"Numeric statistics:\n{stats.to_markdown()}")
        except Exception:
            parts.append(f"Numeric statistics:\n{stats.to_string()}")

    # Categorical top values
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    if cat_cols:
        cat_lines = [
            f"  {c}: {df[c].value_counts().head(5).to_dict()}"
            for c in cat_cols[:10]
        ]
        parts.append("Categorical top values:\n" + "\n".join(cat_lines))

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

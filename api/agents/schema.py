"""Shared helpers for mapping post-encoding feature names back to raw input columns.

The training pipeline stores ``feature_columns`` as the post-encoding names
(e.g. ``mainroad_yes``, ``furnishingstatus_unfurnished``) because that's what
the model was actually fit on. For UX surfaces that ask the user to supply
raw rows (the /predict setup screen), we need the original column names.
"""
from __future__ import annotations


def infer_raw_required_columns(
    feature_cols_post: list[str],
    encoders: dict,
) -> list[str]:
    """Map post-encoding feature names back to raw input column names.

    feature_cols_post are the columns the model was trained on (after
    ``pd.get_dummies`` for one-hot and ``OrdinalEncoder`` for ordinal).
    One-hot columns look like ``{original}_{value}``; ordinal-encoded
    columns keep their original name; numeric columns pass through.
    """
    onehot_sources: list[str]  = list(encoders.get("__onehot__", []) or [])
    ordinal_sources: list[str] = [c for c in encoders.keys() if c != "__onehot__"]

    required: list[str] = []
    for col in feature_cols_post:
        owned_by_onehot = any(col.startswith(f"{src}_") for src in onehot_sources)
        if owned_by_onehot:
            continue
        # Numeric columns and ordinal-encoded columns both keep their original name post-encoding.
        required.append(col)

    for src in onehot_sources:
        if src not in required:
            required.append(src)
    for src in ordinal_sources:
        if src not in required:
            required.append(src)

    seen: set[str] = set()
    deduped: list[str] = []
    for col in required:
        if col not in seen:
            seen.add(col)
            deduped.append(col)
    return deduped

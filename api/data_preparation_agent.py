"""
Automated Data Preparation Agent
---------------------------------
Transforms a raw dataset into train/test splits ready for ML model training.

Pipeline
--------
1.  Drop unusable columns  (constants, IDs, >50 % missing)
2.  Fill missing values    (median for numeric, mode for categorical)
3.  Encode categoricals    (one-hot ≤10 unique, label otherwise)
4.  Scale numeric features (StandardScaler)
5.  Remove highly-correlated features  (|r| > threshold, keep first)
6.  Train / test split     (stratified when target is categorical)
7.  Return artefacts + human-readable preparation report

Modes
-----
  "cleaning" — steps 1-2 only, original dtypes preserved, no split
  "clean"    — steps 1-7 minus the final split (cleaned+encoded+scaled, no split)
  "full"     — complete pipeline including train/test split  (default)
"""
from __future__ import annotations

import re
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from api.logger import get_logger

log = get_logger(__name__)

# ── JSON serialisation helper ─────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ── Shared pipeline helpers ───────────────────────────────────────────────────

_ID_PATTERN = re.compile(r"(^id$|_id$|^index$|_no$|^no$)", re.I)


def _drop_unusable_columns(
    df: pd.DataFrame,
    protect: str | None,
    log: Any,
    steps: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop constant columns, obvious ID columns, and columns with >50 % missing.
    *protect* is a column name that must never be dropped (the target column).
    Returns (cleaned_df, dropped_names).
    """
    dropped: list[str] = []
    keep = lambda c: c != protect and c not in dropped  # noqa: E731

    # Constant columns
    const = [c for c in df.columns if keep(c) and df[c].nunique(dropna=False) <= 1]
    dropped += const
    if const:
        log(f"### Drop constant columns\nDropped: {const}")
        steps.append(f"Dropped constant columns: {const}")

    # ID-like columns (all-unique values, name matches pattern)
    id_cols = [
        c for c in df.columns
        if keep(c) and _ID_PATTERN.search(c) and df[c].nunique() == len(df)
    ]
    dropped += id_cols
    if id_cols:
        log(f"### Drop ID columns\nDropped: {id_cols}")
        steps.append(f"Dropped ID columns: {id_cols}")

    # High-missing columns (>50 %)
    missing_rate = df.isnull().mean()
    high_missing = [c for c in df.columns if keep(c) and missing_rate[c] > 0.50]
    dropped += high_missing
    if high_missing:
        log(f"### Drop high-missing columns (>50 %)\nDropped: {high_missing}")
        steps.append(f"Dropped high-missing columns: {high_missing}")

    return df.drop(columns=dropped), dropped


def _fill_missing(
    df: pd.DataFrame,
    log: Any,
    steps: list[str],
) -> pd.DataFrame:
    """
    Fill NaN values in-place-style (returns new df):
      numeric  → median
      non-numeric → mode (or "Unknown" if mode is empty)
    """
    filled: list[str] = []
    df = df.copy()
    for c in df.columns:
        n = df[c].isnull().sum()
        if n == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            m = df[c].mode()
            df[c] = df[c].fillna(m.iloc[0] if not m.empty else "Unknown")
        filled.append(f"{c} ({n} missing)")

    if filled:
        log(f"  Filled: {filled}")
        steps.append(f"Filled missing values in: {filled}")
    else:
        log("  No missing values found")
        steps.append("No missing values")
    return df


# ── Mode implementations ──────────────────────────────────────────────────────

def _run_cleaning_mode(
    df: pd.DataFrame,
    original_shape: tuple,
    log: Any,
    steps: list[str],
) -> dict:
    """Cleaning-only mode: drop bad columns, fill missing, drop duplicates."""
    log("## Data Cleaning Report")
    log(f"**Original shape:** {original_shape[0]:,} rows × {original_shape[1]} columns\n")

    df, dropped = _drop_unusable_columns(df, protect=None, log=_log, steps=steps)
    log.info("  dropped %d unusable columns: %s", len(dropped), dropped)

    _log("\n### Fill missing values")
    df = _fill_missing(df, _log, steps)

    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates()
        log(f"\n### Drop {n_dups} duplicate rows")
        steps.append(f"Dropped {n_dups} duplicate rows")
    else:
        log("\n### No duplicate rows found")

    log("\n---\n## Summary")
    log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
    log(f"- **After cleaning:** {len(df):,} rows × {df.shape[1]} cols")
    log(f"- **Columns dropped:** {len(dropped)}")
    log("- **No encoding, no scaling — original data types preserved ✓**")

    return {
        "success": True, "mode": "cleaning",
        "target_column": "", "target_type": "",
        "label_mappings": {}, "target_label_map": None,
        "feature_names": df.columns.tolist(),
        "X_train": df.reset_index(drop=True).to_dict(orient="records"),
        "X_test": [], "y_train": [], "y_test": [],
        "train_rows": len(df), "test_rows": 0, "n_features": df.shape[1],
        "dropped_columns": dropped, "corr_dropped": [],
        "encoded_columns": [], "scaled_columns": [],
    }


def _run_full_pipeline(
    df: pd.DataFrame,
    original_shape: tuple,
    target_column: str | None,
    test_size: float,
    random_state: int,
    scale: bool,
    correlation_threshold: float,
    mode: str,
    log: Any,
    steps: list[str],
) -> dict:
    """Clean + encode + scale + (optionally) split."""
    log("## Automated Data Preparation Report")
    log(f"**Original shape:** {original_shape[0]:,} rows × {original_shape[1]} columns\n")

    # Resolve target
    target = (
        target_column
        if target_column and target_column in df.columns
        else df.columns[-1]
    )
    if not (target_column and target_column in df.columns):
        log(f"> No target specified — using last column **`{target}`** as target.\n")
    log(f"**Target column:** `{target}`")
    steps.append(f"Target column: `{target}`")

    # Drop unusable features (never drop target)
    df, dropped = _drop_unusable_columns(df, protect=target, log=log, steps=steps)
    if not dropped:
        steps.append("No unusable columns found")

    # Separate features / target
    X = df.drop(columns=[target])
    y = df[target].copy()

    # Impute
    log("\n### Impute missing values")
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    imputed_num = [c for c in num_cols if X[c].isnull().any()]
    imputed_cat = [c for c in cat_cols if X[c].isnull().any()]
    for c in imputed_num:
        X[c] = X[c].fillna(X[c].median())
    for c in imputed_cat:
        X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")
    missing_target = y.isnull().sum()
    if missing_target > 0:
        y = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else
                     (y.mode().iloc[0] if not y.mode().empty else "Unknown"))
        log(f"  - Imputed {missing_target} missing values in target")
    if imputed_num or imputed_cat:
        log(f"  - Numeric (median): {imputed_num}")
        log(f"  - Categorical (mode): {imputed_cat}")
        steps.append(f"Imputed missing in {len(imputed_num) + len(imputed_cat)} columns")
    else:
        log("  - No missing values found")
        steps.append("No missing values to impute")

    # Encode categoricals
    log("\n### Encode categorical features")
    encoded_cols: list[str] = []
    label_mappings: dict[str, dict] = {}
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    ohe_cols = [c for c in cat_cols if X[c].nunique() <= 10]
    le_cols  = [c for c in cat_cols if X[c].nunique() > 10]
    if ohe_cols:
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=False, dtype=int)
        encoded_cols += ohe_cols
        log(f"  - One-hot encoded (≤10 unique): {ohe_cols}")
    for c in le_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        label_mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
        encoded_cols.append(c)
    if le_cols:
        log(f"  - Label encoded (>10 unique): {le_cols}")
    steps.append(
        f"Encoded {len(encoded_cols)} categorical columns"
        if encoded_cols else "No categorical features to encode"
    )

    # Encode target if categorical
    target_type = "numeric"
    target_label_map: dict | None = None
    if not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), name=target)
        target_label_map = {str(cls): int(i) for i, cls in enumerate(le_y.classes_)}
        target_type = "categorical"
        log(f"  - Target encoded: {target_label_map}")
        steps.append(f"Encoded categorical target ({len(target_label_map)} classes)")

    # Scale
    log("\n### Scale numeric features")
    scaled_cols: list[str] = []
    if scale:
        num_now = X.select_dtypes(include="number").columns.tolist()
        if num_now:
            X[num_now] = StandardScaler().fit_transform(X[num_now])
            scaled_cols = num_now
            log(f"  - StandardScaler applied to {len(num_now)} columns")
            steps.append(f"Scaled {len(num_now)} numeric features")
        else:
            log("  - No numeric features to scale")
    else:
        log("  - Scaling skipped")

    # Drop highly-correlated features
    log(f"\n### Remove highly correlated features (threshold={correlation_threshold})")
    corr_dropped: list[str] = []
    num_now = X.select_dtypes(include="number").columns.tolist()
    if len(num_now) > 1:
        upper = (
            X[num_now].corr().abs()
            .where(np.triu(np.ones((len(num_now), len(num_now))), k=1).astype(bool))
        )
        to_drop = [c for c in upper.columns if upper[c].gt(correlation_threshold).any()]
        if to_drop:
            X = X.drop(columns=to_drop)
            corr_dropped = to_drop
            log(f"  - Dropped: {to_drop}")
            steps.append(f"Dropped {len(to_drop)} highly correlated features")
        else:
            log("  - No highly correlated features found")
    else:
        log("  - Skipped (fewer than 2 numeric features)")

    # Clean-only — return without splitting
    if mode == "clean":
        log("\n### Clean-only mode (no train/test split)")
        steps.append(f"Cleaned: {len(X):,} rows × {X.shape[1]} features (no split)")
        log("\n---\n## Summary")
        log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
        log(f"- **Features after cleaning:** {X.shape[1]}")
        log(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
        log("- **Ready for exploration / feature engineering ✓**")
        return {
            "success": True, "mode": "clean",
            "target_column": "", "target_type": "",
            "label_mappings": label_mappings, "target_label_map": None,
            "feature_names": X.columns.tolist(),
            "X_train": X.reset_index(drop=True).to_dict(orient="records"),
            "X_test": [], "y_train": [], "y_test": [],
            "train_rows": len(X), "test_rows": 0, "n_features": X.shape[1],
            "dropped_columns": dropped, "corr_dropped": corr_dropped,
            "encoded_columns": encoded_cols, "scaled_columns": scaled_cols,
        }

    # Full — train/test split
    log(f"\n### Train / test split ({int((1 - test_size) * 100)}/{int(test_size * 100)})")
    stratify = y if target_type == "categorical" and y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify,
    )
    log(f"  - Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    steps.append(f"Split: {len(X_train):,} train / {len(X_test):,} test rows")

    log("\n---\n## Summary")
    log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
    log(f"- **Features after prep:** {X_train.shape[1]}")
    log(f"- **Target:** `{target}` ({target_type})")
    log(f"- **Train rows:** {len(X_train):,} | **Test rows:** {len(X_test):,}")
    log(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
    log("- **Ready for model training ✓**")

    return {
        "success": True, "mode": "full",
        "target_column": target, "target_type": target_type,
        "label_mappings": label_mappings, "target_label_map": target_label_map,
        "feature_names": X_train.columns.tolist(),
        "X_train": X_train.reset_index(drop=True).to_dict(orient="records"),
        "X_test":  X_test.reset_index(drop=True).to_dict(orient="records"),
        "y_train": y_train.reset_index(drop=True).tolist(),
        "y_test":  y_test.reset_index(drop=True).tolist(),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "n_features": X_train.shape[1],
        "dropped_columns": dropped, "corr_dropped": corr_dropped,
        "encoded_columns": encoded_cols, "scaled_columns": scaled_cols,
    }


# ── Public entry point ────────────────────────────────────────────────────────

def run_data_preparation(
    data: list[dict],
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    correlation_threshold: float = 0.95,
    mode: str = "full",
) -> dict:
    """
    Run the data-preparation pipeline.

    Parameters
    ----------
    data                  : raw dataset rows (list of dicts)
    target_column         : label column name; defaults to the last column
    test_size             : test-split fraction  (default 0.20)
    random_state          : reproducibility seed
    scale                 : apply StandardScaler to numeric features
    correlation_threshold : drop one of any pair with |r| > this value
    mode                  : "full" | "clean" | "cleaning"
    """
    import time
    t0 = time.perf_counter()

    report_lines: list[str] = []
    steps:        list[str] = []

    def _log(msg: str) -> None:
        report_lines.append(msg)

    log.info("━━ DataPrep start  mode='%s'  rows=%d ━━", mode, len(data))

    try:
        df = pd.DataFrame(data)
        original_shape = df.shape
        log.info("  loaded DataFrame  shape=%s", original_shape)

        if mode == "cleaning":
            result = _run_cleaning_mode(df, original_shape, _log, steps)
        else:
            result = _run_full_pipeline(
                df, original_shape,
                target_column, test_size, random_state,
                scale, correlation_threshold, mode,
                _log, steps,
            )

        result["report"] = "\n".join(report_lines)
        result["steps"]  = steps

        log.info(
            "━━ DataPrep done  mode='%s'  train=%d  test=%d  features=%d  elapsed=%.1fs ━━",
            result.get("mode"), result.get("train_rows", 0),
            result.get("test_rows", 0), result.get("n_features", 0),
            time.perf_counter() - t0,
        )
        return _json_safe(result)

    except Exception as exc:
        log.exception("━━ DataPrep FAILED  elapsed=%.1fs: %s ━━", time.perf_counter() - t0, exc)
        return {
            "success":   False,
            "error":     str(exc),
            "traceback": traceback.format_exc(),
            "report":    f"## Data Preparation Failed\n\n**Error:** {exc}",
            "steps":     steps,
        }

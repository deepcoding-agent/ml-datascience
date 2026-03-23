"""
Automated Data Preparation Agent  (Sprint 1 upgrade)
-----------------------------------------------------
Transforms a raw dataset into train/test splits ready for ML model training.

Pipeline (full mode)
--------------------
1.  Infer data types     (detect numeric/datetime stored as strings)
2.  Drop unusable columns (constants, IDs, >drop_threshold missing)
3.  Fill missing values   (auto: skewed→median, normal→mean, cat→mode)
4.  Remove duplicate rows
5.  Detect & clip outliers (IQR or z-score, configurable)
6.  Encode categoricals   (auto/onehot/label/ordinal)
7.  Scale numeric features (standard/minmax/robust/none)
8.  Remove highly-correlated features (|r| > threshold)
9.  Train / test split    (stratified when target is categorical)
10. Return artefacts + structured preprocessing report

Modes
-----
  "cleaning" — steps 1-4 only, original dtypes preserved, no encoding/scaling/split
  "clean"    — steps 1-8, no train/test split
  "full"     — complete pipeline including train/test split  (default)
"""
from __future__ import annotations

import re
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from api.logger import get_logger
from api.models import PrepConfig, PrepReportDetail

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


# ── Constants ────────────────────────────────────────────────────────────────

_ID_PATTERN = re.compile(r"(^id$|_id$|^index$|_no$|^no$)", re.I)


# ── Step 1: Data type inference ──────────────────────────────────────────────


def _infer_dtypes(
    df: pd.DataFrame,
    report: PrepReportDetail,
    lines: list[str],
) -> pd.DataFrame:
    """Detect string columns that should be numeric or datetime and convert."""
    df = df.copy()
    inferred: dict[str, str] = {}

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(200)
        if sample.empty:
            continue

        # Try numeric
        coerced = pd.to_numeric(sample, errors="coerce")
        if coerced.notna().mean() >= 0.8:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            inferred[col] = "numeric"
            continue

        # Try datetime
        try:
            coerced_dt = pd.to_datetime(sample, errors="coerce", format="mixed")
            if coerced_dt.notna().mean() >= 0.8:
                df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
                inferred[col] = "datetime"
        except Exception:
            pass

    if inferred:
        lines.append(f"### Infer data types\n  Converted: {inferred}")
        report.dtype_inferred = inferred
        report.steps_applied.append(f"Inferred dtypes for {len(inferred)} columns")
    return df


# ── Step 2: Drop unusable columns ───────────────────────────────────────────


def _drop_unusable_columns(
    df: pd.DataFrame,
    protect: str | None,
    drop_threshold: float,
    report: PrepReportDetail,
    lines: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Drop constant, ID-like, and high-missing columns."""
    dropped: list[str] = []
    keep = lambda c: c != protect and c not in dropped  # noqa: E731

    # Constant columns
    const = [c for c in df.columns if keep(c) and df[c].nunique(dropna=False) <= 1]
    dropped += const

    # ID-like columns
    id_cols = [
        c for c in df.columns
        if keep(c) and _ID_PATTERN.search(c) and df[c].nunique() == len(df)
    ]
    dropped += id_cols

    # High-missing columns
    missing_rate = df.isnull().mean()
    high_missing = [c for c in df.columns if keep(c) and missing_rate[c] > drop_threshold]
    dropped += high_missing

    if dropped:
        lines.append(f"### Drop unusable columns\n  Dropped {len(dropped)}: {dropped}")
        report.columns_dropped = dropped
        report.steps_applied.append(f"Dropped {len(dropped)} unusable columns")

    return df.drop(columns=dropped), dropped


# ── Step 3: Smart missing value handling ─────────────────────────────────────


def _fill_missing(
    df: pd.DataFrame,
    strategy: str,
    report: PrepReportDetail,
    lines: list[str],
) -> pd.DataFrame:
    """
    Fill NaN values with smart strategy selection.
      auto   : numeric skewed→median, numeric normal→mean, categorical→mode
      mean   : force mean for numeric
      median : force median for numeric
      mode   : force mode for all
      drop   : drop rows with any NaN
    """
    df = df.copy()
    filled: dict[str, str] = {}

    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        count = before - len(df)
        if count > 0:
            filled["_rows_dropped"] = f"{count} rows"
            lines.append(f"### Fill missing values\n  Dropped {count} rows with NaN")
            report.missing_filled = filled
            report.steps_applied.append(f"Dropped {count} rows with missing values")
        return df

    for col in df.columns:
        n = df[col].isnull().sum()
        if n == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
                filled[col] = "mean"
            elif strategy == "median":
                df[col] = df[col].fillna(df[col].median())
                filled[col] = "median"
            elif strategy == "mode":
                m = df[col].mode()
                df[col] = df[col].fillna(m.iloc[0] if not m.empty else 0)
                filled[col] = "mode"
            else:  # auto
                skewness = df[col].dropna().skew()
                if abs(skewness) > 1.0:
                    df[col] = df[col].fillna(df[col].median())
                    filled[col] = f"median (skew={skewness:.2f})"
                else:
                    df[col] = df[col].fillna(df[col].mean())
                    filled[col] = f"mean (skew={skewness:.2f})"
        else:
            m = df[col].mode()
            df[col] = df[col].fillna(m.iloc[0] if not m.empty else "Unknown")
            filled[col] = "mode"

    if filled:
        lines.append(f"### Fill missing values\n  {filled}")
        report.missing_filled = filled
        report.steps_applied.append(f"Filled missing in {len(filled)} columns")
    else:
        lines.append("### Fill missing values\n  No missing values found")
    return df


# ── Step 4: Duplicate removal ───────────────────────────────────────────────


def _remove_duplicates(
    df: pd.DataFrame,
    report: PrepReportDetail,
    lines: list[str],
) -> pd.DataFrame:
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        df = df.drop_duplicates()
        lines.append(f"### Remove duplicates\n  Dropped {n_dups} duplicate rows")
        report.duplicates_removed = int(n_dups)
        report.steps_applied.append(f"Removed {n_dups} duplicate rows")
    else:
        lines.append("### Remove duplicates\n  No duplicates found")
    return df


# ── Step 5: Outlier detection & clipping ─────────────────────────────────────


def _clip_outliers(
    df: pd.DataFrame,
    method: str,
    threshold: float,
    report: PrepReportDetail,
    lines: list[str],
) -> pd.DataFrame:
    """Clip outliers to bounds instead of removing rows."""
    if method == "none":
        lines.append("### Outlier treatment\n  Skipped (disabled)")
        return df

    df = df.copy()
    clipped: dict[str, int] = {}
    num_cols = df.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        if method == "iqr":
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        elif method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            continue

        mask = (df[col] < lower) | (df[col] > upper)
        count = mask.sum()
        if count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            clipped[col] = int(count)

    if clipped:
        total = sum(clipped.values())
        lines.append(
            f"### Outlier treatment ({method}, threshold={threshold})\n"
            f"  Clipped {total} values across {len(clipped)} columns"
        )
        report.outliers_clipped = clipped
        report.steps_applied.append(
            f"Clipped outliers in {len(clipped)} columns ({total} values)"
        )
    else:
        lines.append(f"### Outlier treatment ({method})\n  No outliers detected")
    return df


# ── Step 6: Encode categoricals ──────────────────────────────────────────────


def _encode_categoricals(
    X: pd.DataFrame,
    method: str,
    report: PrepReportDetail,
    lines: list[str],
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Encode categorical features. Returns (encoded_df, label_mappings)."""
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    # Drop datetime columns from encoding — they shouldn't be one-hot encoded
    cat_cols = [c for c in cat_cols if not pd.api.types.is_datetime64_any_dtype(X[c])]

    if not cat_cols:
        lines.append("### Encode categoricals\n  No categorical features to encode")
        return X, {}

    label_mappings: dict[str, dict] = {}
    encodings_applied: dict[str, str] = {}

    if method == "onehot":
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=int)
        for c in cat_cols:
            encodings_applied[c] = "onehot"
    elif method == "label":
        for c in cat_cols:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            label_mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
            encodings_applied[c] = "label"
    elif method == "ordinal":
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = oe.fit_transform(X[cat_cols].astype(str))
        for c in cat_cols:
            encodings_applied[c] = "ordinal"
    else:  # auto
        ohe_cols = [c for c in cat_cols if X[c].nunique() <= 10]
        le_cols = [c for c in cat_cols if X[c].nunique() > 10]
        if ohe_cols:
            X = pd.get_dummies(X, columns=ohe_cols, drop_first=False, dtype=int)
            for c in ohe_cols:
                encodings_applied[c] = "onehot"
        for c in le_cols:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))
            label_mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
            encodings_applied[c] = "label"

    lines.append(f"### Encode categoricals\n  {encodings_applied}")
    report.encodings_applied = encodings_applied
    report.steps_applied.append(f"Encoded {len(encodings_applied)} categorical columns")
    return X, label_mappings


# ── Step 7: Scale numeric features ──────────────────────────────────────────


def _scale_features(
    X: pd.DataFrame,
    method: str,
    report: PrepReportDetail,
    lines: list[str],
) -> pd.DataFrame:
    if method == "none":
        lines.append("### Scale features\n  Skipped (disabled)")
        return X

    num_cols = X.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        lines.append("### Scale features\n  No numeric features to scale")
        return X

    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    scaler_cls = scalers.get(method, StandardScaler)
    X[num_cols] = scaler_cls().fit_transform(X[num_cols])

    lines.append(f"### Scale features\n  {method} scaler applied to {len(num_cols)} columns")
    report.scaler_used = method
    report.steps_applied.append(f"Scaled {len(num_cols)} features ({method})")
    return X


# ── Step 8: Correlation pruning ──────────────────────────────────────────────


def _drop_correlated(
    X: pd.DataFrame,
    threshold: float,
    report: PrepReportDetail,
    lines: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    corr_dropped: list[str] = []

    if len(num_cols) > 1:
        corr_matrix = X[num_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        to_drop = [c for c in upper.columns if upper[c].gt(threshold).any()]
        if to_drop:
            X = X.drop(columns=to_drop)
            corr_dropped = to_drop
            lines.append(
                f"### Correlation pruning (threshold={threshold})\n"
                f"  Dropped {len(to_drop)}: {to_drop}"
            )
            report.steps_applied.append(
                f"Dropped {len(to_drop)} highly correlated features"
            )
        else:
            lines.append(f"### Correlation pruning\n  No highly correlated features found")
    else:
        lines.append("### Correlation pruning\n  Skipped (fewer than 2 numeric features)")

    return X, corr_dropped


# ── Mode: cleaning only ─────────────────────────────────────────────────────


def _run_cleaning_mode(
    df: pd.DataFrame,
    original_shape: tuple[int, int],
    cfg: PrepConfig,
    report: PrepReportDetail,
    lines: list[str],
) -> dict:
    """Steps 1-4 only: infer types, drop bad columns, fill missing, deduplicate."""
    lines.append("## Data Cleaning Report")
    lines.append(f"**Original shape:** {original_shape[0]:,} rows x {original_shape[1]} columns\n")

    df = _infer_dtypes(df, report, lines)
    df, dropped = _drop_unusable_columns(df, protect=None, drop_threshold=cfg.drop_threshold, report=report, lines=lines)
    df = _fill_missing(df, cfg.missing_strategy, report, lines)
    df = _remove_duplicates(df, report, lines)

    lines.append(f"\n---\n## Summary")
    lines.append(f"- **Original:** {original_shape[0]:,} rows x {original_shape[1]} cols")
    lines.append(f"- **After cleaning:** {len(df):,} rows x {df.shape[1]} cols")
    lines.append(f"- **Columns dropped:** {len(dropped)}")
    lines.append("- **No encoding, no scaling — original data types preserved**")

    report.rows_after = len(df)
    report.columns_after = df.shape[1]

    return {
        "success": True, "mode": "cleaning",
        "target_column": "", "target_type": "",
        "label_mappings": {}, "target_label_map": None,
        "feature_names": df.columns.tolist(),
        "X_train": df.reset_index(drop=True).to_dict(orient="records"),
        "X_test": [], "y_train": [], "y_test": [],
        "train_rows": len(df), "test_rows": 0, "n_features": df.shape[1],
    }


# ── Mode: clean / full ──────────────────────────────────────────────────────


def _run_full_pipeline(
    df: pd.DataFrame,
    original_shape: tuple[int, int],
    target_column: str | None,
    cfg: PrepConfig,
    mode: str,
    report: PrepReportDetail,
    lines: list[str],
) -> dict:
    """Full pipeline: infer → drop → fill → dedup → outliers → encode → scale → corr → split."""
    lines.append("## Automated Data Preparation Report")
    lines.append(f"**Original shape:** {original_shape[0]:,} rows x {original_shape[1]} columns\n")

    # Step 1: Infer dtypes
    df = _infer_dtypes(df, report, lines)

    # Resolve target
    target = (
        target_column
        if target_column and target_column in df.columns
        else df.columns[-1]
    )
    if not (target_column and target_column in df.columns):
        lines.append(f"> No target specified — using last column **`{target}`** as target.\n")
    lines.append(f"**Target column:** `{target}`")
    report.steps_applied.append(f"Target column: {target}")

    # Step 2: Drop unusable
    df, dropped = _drop_unusable_columns(df, protect=target, drop_threshold=cfg.drop_threshold, report=report, lines=lines)

    # Separate features / target
    X = df.drop(columns=[target])
    y = df[target].copy()

    # Step 3: Fill missing
    X = _fill_missing(X, cfg.missing_strategy, report, lines)
    missing_target = y.isnull().sum()
    if missing_target > 0:
        y = y.fillna(
            y.median() if pd.api.types.is_numeric_dtype(y)
            else (y.mode().iloc[0] if not y.mode().empty else "Unknown")
        )
        lines.append(f"  - Imputed {missing_target} missing values in target")

    # Step 4: Remove duplicates (on features, keep corresponding y)
    dup_mask = X.duplicated()
    n_dups = dup_mask.sum()
    if n_dups > 0:
        X = X[~dup_mask]
        y = y[X.index]
        lines.append(f"### Remove duplicates\n  Dropped {n_dups} duplicate rows")
        report.duplicates_removed = int(n_dups)
        report.steps_applied.append(f"Removed {n_dups} duplicate rows")

    # Step 5: Outlier clipping (numeric features only)
    X = _clip_outliers(X, cfg.outlier_treatment, cfg.outlier_threshold, report, lines)

    # Step 6: Encode categoricals
    X, label_mappings = _encode_categoricals(X, cfg.encoding_method, report, lines)

    # Encode target if categorical
    target_type = "numeric"
    target_label_map: dict | None = None
    if not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), name=target, index=y.index)
        target_label_map = {str(cls): int(i) for i, cls in enumerate(le_y.classes_)}
        target_type = "categorical"
        lines.append(f"  - Target encoded: {target_label_map}")
        report.steps_applied.append(f"Encoded categorical target ({len(target_label_map)} classes)")

    # Drop datetime columns (not useful for ML after this point)
    dt_cols = X.select_dtypes(include="datetime").columns.tolist()
    if dt_cols:
        X = X.drop(columns=dt_cols)
        lines.append(f"### Drop datetime columns\n  Dropped (not ML-ready): {dt_cols}")
        report.steps_applied.append(f"Dropped {len(dt_cols)} datetime columns")

    # Step 7: Scale
    X = _scale_features(X, cfg.scaling_method, report, lines)

    # Step 8: Correlation pruning
    X, corr_dropped = _drop_correlated(X, cfg.correlation_threshold, report, lines)

    report.rows_after = len(X)
    report.columns_after = X.shape[1]

    # Clean-only mode — no split
    if mode == "clean":
        lines.append("\n### Clean-only mode (no train/test split)")
        lines.append(f"\n---\n## Summary")
        lines.append(f"- **Original:** {original_shape[0]:,} rows x {original_shape[1]} cols")
        lines.append(f"- **Features after cleaning:** {X.shape[1]}")
        lines.append(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
        lines.append("- **Ready for exploration / feature engineering**")
        return {
            "success": True, "mode": "clean",
            "target_column": "", "target_type": "",
            "label_mappings": label_mappings, "target_label_map": None,
            "feature_names": X.columns.tolist(),
            "X_train": X.reset_index(drop=True).to_dict(orient="records"),
            "X_test": [], "y_train": [], "y_test": [],
            "train_rows": len(X), "test_rows": 0, "n_features": X.shape[1],
        }

    # Step 9: Train/test split
    split_pct = f"{int((1 - cfg.test_size) * 100)}/{int(cfg.test_size * 100)}"
    lines.append(f"\n### Train / test split ({split_pct})")
    stratify = y if target_type == "categorical" and y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )
    lines.append(f"  - Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    report.steps_applied.append(f"Split: {len(X_train):,} train / {len(X_test):,} test")

    lines.append(f"\n---\n## Summary")
    lines.append(f"- **Original:** {original_shape[0]:,} rows x {original_shape[1]} cols")
    lines.append(f"- **Features after prep:** {X_train.shape[1]}")
    lines.append(f"- **Target:** `{target}` ({target_type})")
    lines.append(f"- **Train rows:** {len(X_train):,} | **Test rows:** {len(X_test):,}")
    lines.append(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
    lines.append("- **Ready for model training**")

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
    }


# ── Public entry point ───────────────────────────────────────────────────────


def run_data_preparation(
    data: list[dict],
    target_column: str | None = None,
    mode: str = "full",
    config: PrepConfig | None = None,
) -> dict:
    """
    Run the data-preparation pipeline.

    Parameters
    ----------
    data           : raw dataset rows (list of dicts)
    target_column  : label column name; defaults to the last column
    mode           : "full" | "clean" | "cleaning"
    config         : PrepConfig with all tuneable parameters
    """
    t0 = time.perf_counter()
    cfg = config or PrepConfig()

    report = PrepReportDetail()
    report_lines: list[str] = []

    log.info("━━ DataPrep start  mode='%s'  rows=%d ━━", mode, len(data))

    try:
        df = pd.DataFrame(data)
        original_shape = df.shape
        report.rows_before = original_shape[0]
        report.columns_before = original_shape[1]
        log.info("  loaded DataFrame  shape=%s", original_shape)

        if mode == "cleaning":
            result = _run_cleaning_mode(df, original_shape, cfg, report, report_lines)
        else:
            result = _run_full_pipeline(
                df, original_shape, target_column, cfg, mode, report, report_lines,
            )

        elapsed = time.perf_counter() - t0
        report.duration_seconds = round(elapsed, 2)
        result["report"] = "\n".join(report_lines)
        result["report_detail"] = report.model_dump()

        log.info(
            "━━ DataPrep done  mode='%s'  train=%d  test=%d  features=%d  elapsed=%.1fs ━━",
            result.get("mode"), result.get("train_rows", 0),
            result.get("test_rows", 0), result.get("n_features", 0), elapsed,
        )
        return _json_safe(result)

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        report.duration_seconds = round(elapsed, 2)
        log.exception("━━ DataPrep FAILED  elapsed=%.1fs: %s ━━", elapsed, exc)
        return {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "report": f"## Data Preparation Failed\n\n**Error:** {exc}",
            "report_detail": report.model_dump(),
        }

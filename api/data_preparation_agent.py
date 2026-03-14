"""
Automated Data Preparation Agent
---------------------------------
Transforms a raw dataset into train/test splits ready for ML model training.

Pipeline
--------
1.  Drop columns that are unusable (IDs, constants, >50% missing)
2.  Handle missing values  (median for numeric, mode for categorical)
3.  Encode categorical features  (one-hot ≤10 unique, label otherwise)
4.  Scale numeric features  (StandardScaler)
5.  Remove highly-correlated features  (|r| > 0.95, keep first)
6.  Train / test split  (80/20, stratified when target is categorical)
7.  Return artefacts + human-readable preparation report
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── helpers ──────────────────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


# ── main agent ───────────────────────────────────────────────────────────────

def run_data_preparation(
    data: list[dict],
    target_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    correlation_threshold: float = 0.95,
    mode: str = "full",  # "full" | "clean" | "cleaning"
) -> dict:
    """
    Run the full data-preparation pipeline.

    Parameters
    ----------
    data            : raw dataset rows (list of dicts from frontend)
    target_column   : name of the label / target column.
                      If None, the last column is used automatically.
    test_size       : fraction used for the test split (default 0.20)
    random_state    : reproducibility seed
    scale           : whether to apply StandardScaler to numeric features
    correlation_threshold : drop one of any pair with |r| > this value

    Returns
    -------
    dict with keys:
        success, report, steps,
        feature_names,
        X_train, X_test, y_train, y_test,
        train_rows, test_rows, n_features,
        dropped_columns, encoded_columns, scaled_columns, corr_dropped,
        target_column, target_type,
        label_mappings   (only for encoded categorical target)
    """
    report_lines: list[str] = []
    steps: list[str] = []

    def log(msg: str) -> None:
        report_lines.append(msg)

    try:
        df = pd.DataFrame(data)
        original_shape = df.shape

        # ══ CLEANING MODE — data quality only, no encoding / scaling / split ══
        if mode == "cleaning":
            log(f"## Data Cleaning Report")
            log(f"**Original shape:** {original_shape[0]:,} rows × {original_shape[1]} columns\n")

            dropped: list[str] = []

            # Step 1 — Drop constant columns
            const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
            dropped += const_cols
            if const_cols:
                log(f"### Step 1 — Drop constant columns\nDropped: {const_cols}")
                steps.append(f"Dropped constant columns: {const_cols}")

            # Step 2 — Drop obvious ID columns
            id_pattern = re.compile(r"(^id$|_id$|^index$|_no$|^no$)", re.I)
            id_cols = [
                c for c in df.columns
                if c not in dropped
                and id_pattern.search(c)
                and df[c].nunique() == len(df)
            ]
            dropped += id_cols
            if id_cols:
                log(f"### Step 2 — Drop ID columns\nDropped: {id_cols}")
                steps.append(f"Dropped ID columns: {id_cols}")

            # Step 3 — Drop columns with >50% missing
            missing_rate = df.isnull().mean()
            high_missing = [c for c in df.columns if c not in dropped and missing_rate[c] > 0.50]
            dropped += high_missing
            if high_missing:
                log(f"### Step 3 — Drop high-missing columns (>50%)\nDropped: {high_missing}")
                steps.append(f"Dropped high-missing columns: {high_missing}")

            df = df.drop(columns=dropped)

            # Step 4 — Fill missing values (keep original dtypes)
            log(f"\n### Step 4 — Fill missing values")
            filled: list[str] = []
            for c in df.columns:
                n_missing = df[c].isnull().sum()
                if n_missing == 0:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
                else:
                    mode_val = df[c].mode()
                    df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
                filled.append(f"{c} ({n_missing} missing)")
            if filled:
                log(f"  Filled: {filled}")
                steps.append(f"Filled missing values in: {filled}")
            else:
                log("  No missing values found")
                steps.append("No missing values")

            # Step 5 — Drop duplicate rows
            n_dups = df.duplicated().sum()
            if n_dups > 0:
                df = df.drop_duplicates()
                log(f"\n### Step 5 — Drop {n_dups} duplicate rows")
                steps.append(f"Dropped {n_dups} duplicate rows")
            else:
                log(f"\n### Step 5 — No duplicate rows found")

            log(f"\n---\n## Summary")
            log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
            log(f"- **After cleaning:** {len(df):,} rows × {df.shape[1]} cols")
            log(f"- **Columns dropped:** {len(dropped)}")
            log(f"- **No encoding, no scaling — original data types preserved ✓**")

            return _json_safe({
                "success": True,
                "mode": "cleaning",
                "report": "\n".join(report_lines),
                "steps": steps,
                "target_column": "",
                "target_type": "",
                "label_mappings": {},
                "target_label_map": None,
                "feature_names": df.columns.tolist(),
                "X_train": df.reset_index(drop=True).to_dict(orient="records"),
                "X_test": [],
                "y_train": [],
                "y_test": [],
                "train_rows": len(df),
                "test_rows": 0,
                "n_features": df.shape[1],
                "dropped_columns": dropped,
                "corr_dropped": [],
                "encoded_columns": [],
                "scaled_columns": [],
            })
        # ════════════════════════════════════════════════════════════════════

        log(f"## Automated Data Preparation Report")
        log(f"**Original shape:** {original_shape[0]:,} rows × {original_shape[1]} columns\n")

        # ── 1. Resolve target column ──────────────────────────────────────
        if target_column and target_column in df.columns:
            target = target_column
        else:
            target = df.columns[-1]
            log(f"> No target specified — using last column **`{target}`** as target.\n")
        steps.append(f"Target column: `{target}`")
        log(f"**Target column:** `{target}`")

        # ── 2. Drop unusable columns ──────────────────────────────────────
        dropped: list[str] = []

        # Constant columns
        const_cols = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
        dropped += const_cols
        if const_cols:
            log(f"\n### Step 1 — Drop constant columns\nDropped: {const_cols}")

        # Likely ID columns (integer, all unique, name contains 'id' / 'index' / '_no')
        id_pattern = re.compile(r"(^id$|_id$|^index$|_no$|^no$)", re.I)
        id_cols = [
            c for c in df.columns
            if c != target
            and c not in dropped
            and id_pattern.search(c)
            and df[c].nunique() == len(df)
        ]
        dropped += id_cols
        if id_cols:
            log(f"\n### Step 2 — Drop ID columns\nDropped: {id_cols}")

        # High-missing columns (>50%)
        missing_rate = df.isnull().mean()
        high_missing = [
            c for c in df.columns
            if c != target and c not in dropped and missing_rate[c] > 0.50
        ]
        dropped += high_missing
        if high_missing:
            log(f"\n### Step 3 — Drop high-missing columns (>50%)\nDropped: {high_missing}")

        df = df.drop(columns=dropped)
        if dropped:
            steps.append(f"Dropped {len(dropped)} unusable columns: {dropped}")
        else:
            steps.append("No unusable columns found")

        # ── 3. Separate features / target ─────────────────────────────────
        X = df.drop(columns=[target])
        y = df[target].copy()

        # ── 4. Impute missing values ──────────────────────────────────────
        log(f"\n### Step 4 — Impute missing values")
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()

        imputed_num = [c for c in num_cols if X[c].isnull().any()]
        imputed_cat = [c for c in cat_cols if X[c].isnull().any()]

        for c in imputed_num:
            X[c] = X[c].fillna(X[c].median())
        for c in imputed_cat:
            X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else "Unknown")

        # Handle missing target
        missing_target = y.isnull().sum()
        if missing_target > 0:
            if pd.api.types.is_numeric_dtype(y):
                y = y.fillna(y.median())
            else:
                y = y.fillna(y.mode().iloc[0] if not y.mode().empty else "Unknown")
            log(f"  - Imputed {missing_target} missing values in target")

        if imputed_num or imputed_cat:
            log(f"  - Numeric (median): {imputed_num}")
            log(f"  - Categorical (mode): {imputed_cat}")
            steps.append(f"Imputed missing values in {len(imputed_num)+len(imputed_cat)} columns")
        else:
            log("  - No missing values found")
            steps.append("No missing values to impute")

        # ── 5. Encode categorical features ────────────────────────────────
        log(f"\n### Step 5 — Encode categorical features")
        encoded_cols: list[str] = []
        label_mappings: dict[str, dict] = {}

        # Re-detect after imputation
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

        # Encode target if categorical
        target_type = "numeric"
        target_label_map: dict | None = None
        if not pd.api.types.is_numeric_dtype(y):
            le_y = LabelEncoder()
            y_encoded = le_y.fit_transform(y.astype(str))
            target_label_map = {str(cls): int(i) for i, cls in enumerate(le_y.classes_)}
            y = pd.Series(y_encoded, name=target)
            target_type = "categorical"
            log(f"  - Target encoded: {target_label_map}")
            steps.append(f"Encoded categorical target ({len(target_label_map)} classes)")

        if encoded_cols:
            steps.append(f"Encoded {len(encoded_cols)} categorical feature columns")
        else:
            steps.append("No categorical features to encode")

        # ── 6. Scale numeric features ──────────────────────────────────────
        log(f"\n### Step 6 — Scale numeric features")
        scaled_cols: list[str] = []

        if scale:
            num_cols_now = X.select_dtypes(include="number").columns.tolist()
            if num_cols_now:
                scaler = StandardScaler()
                X[num_cols_now] = scaler.fit_transform(X[num_cols_now])
                scaled_cols = num_cols_now
                log(f"  - StandardScaler applied to {len(num_cols_now)} numeric columns")
                steps.append(f"Scaled {len(num_cols_now)} numeric features (StandardScaler)")
            else:
                log("  - No numeric features to scale")
        else:
            log("  - Scaling skipped")

        # ── 7. Remove highly correlated features ──────────────────────────
        log(f"\n### Step 7 — Remove highly correlated features (threshold={correlation_threshold})")
        corr_dropped: list[str] = []

        num_cols_now = X.select_dtypes(include="number").columns.tolist()
        if len(num_cols_now) > 1:
            corr_matrix = X[num_cols_now].corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
            if to_drop:
                X = X.drop(columns=to_drop)
                corr_dropped = to_drop
                log(f"  - Dropped highly correlated features: {to_drop}")
                steps.append(f"Dropped {len(to_drop)} highly correlated features")
            else:
                log("  - No highly correlated features found")
        else:
            log("  - Skipped (fewer than 2 numeric features)")

        # ── 8a. Clean-only mode — return cleaned dataset, skip split ──────
        if mode == "clean":
            log(f"\n### Step 8 — Clean-only mode (no train/test split)")
            log(f"  - Returning cleaned dataset: {len(X):,} rows × {X.shape[1]} features")
            steps.append(f"Cleaned dataset: {len(X):,} rows × {X.shape[1]} features (no split)")
            log(f"\n---\n## Summary")
            log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
            log(f"- **Features after cleaning:** {X.shape[1]}")
            log(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
            log(f"- **Ready for exploration / feature engineering ✓**")
            return _json_safe({
                "success": True,
                "mode": "clean",
                "report": "\n".join(report_lines),
                "steps": steps,
                "target_column": "",
                "target_type": "",
                "label_mappings": label_mappings,
                "target_label_map": None,
                "feature_names": X.columns.tolist(),
                "X_train": X.reset_index(drop=True).to_dict(orient="records"),
                "X_test": [],
                "y_train": [],
                "y_test": [],
                "train_rows": len(X),
                "test_rows": 0,
                "n_features": X.shape[1],
                "dropped_columns": dropped,
                "corr_dropped": corr_dropped,
                "encoded_columns": encoded_cols,
                "scaled_columns": scaled_cols,
            })

        # ── 8b. Full mode — Train / test split ────────────────────────────
        log(f"\n### Step 8 — Train / test split ({int((1-test_size)*100)}/{int(test_size*100)})")

        stratify = y if target_type == "categorical" and y.nunique() <= 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
        log(f"  - Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
        log(f"  - Features: {X_train.shape[1]}")
        steps.append(f"Split: {len(X_train):,} train / {len(X_test):,} test rows")

        # ── 9. Final summary ───────────────────────────────────────────────
        log(f"\n---\n## Summary")
        log(f"- **Original:** {original_shape[0]:,} rows × {original_shape[1]} cols")
        log(f"- **Features after prep:** {X_train.shape[1]}")
        log(f"- **Target:** `{target}` ({target_type})")
        log(f"- **Train rows:** {len(X_train):,} | **Test rows:** {len(X_test):,}")
        log(f"- **Dropped columns:** {len(dropped) + len(corr_dropped)}")
        log(f"- **Ready for model training ✓**")

        return _json_safe({
            "success": True,
            "mode": "full",
            "report": "\n".join(report_lines),
            "steps": steps,
            "target_column": target,
            "target_type": target_type,
            "label_mappings": label_mappings,
            "target_label_map": target_label_map,
            "feature_names": X_train.columns.tolist(),
            "X_train": X_train.reset_index(drop=True).to_dict(orient="records"),
            "X_test":  X_test.reset_index(drop=True).to_dict(orient="records"),
            "y_train": y_train.reset_index(drop=True).tolist(),
            "y_test":  y_test.reset_index(drop=True).tolist(),
            "train_rows": len(X_train),
            "test_rows":  len(X_test),
            "n_features": X_train.shape[1],
            "dropped_columns": dropped,
            "corr_dropped": corr_dropped,
            "encoded_columns": encoded_cols,
            "scaled_columns": scaled_cols,
        })

    except Exception as exc:
        import traceback
        return {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "report": f"## Data Preparation Failed\n\n**Error:** {exc}",
            "steps": steps,
        }

"""PredictAgent — load a saved model and produce predictions on new rows.

Reverses the preprocessing performed by train_agent._preprocess so the
incoming raw rows are transformed with the exact same scaler / encoders
that produced the training feature space.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.agents.model_storage import load_model
from api.logger import get_logger

log = get_logger(__name__)


def _infer_raw_required_columns(
    feature_cols_post: list[str],
    encoders: dict,
) -> list[str]:
    """Best-effort: figure out which raw input columns the model expects.

    feature_cols_post are the post-encoding feature names. One-hot columns
    look like ``{original}_{value}``; ordinal-encoded columns keep their
    original name. We collapse one-hot outputs back to their original.
    """
    onehot_sources: list[str] = list(encoders.get("__onehot__", []) or [])
    ordinal_sources: list[str] = [c for c in encoders.keys() if c != "__onehot__"]

    required: list[str] = []
    for col in feature_cols_post:
        owned_by_onehot = any(col.startswith(f"{src}_") for src in onehot_sources)
        if owned_by_onehot:
            continue
        if col in ordinal_sources:
            required.append(col)
            continue
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


def _apply_preprocessing(
    df: pd.DataFrame,
    pipeline: dict,
    feature_cols_post: list[str],
) -> pd.DataFrame:
    """Apply the same fillna / one-hot / ordinal / scaling as training."""
    encoders = pipeline.get("encoders", {}) or {}

    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].isnull().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "missing")

    onehot_cols = list(encoders.get("__onehot__", []) or [])
    onehot_in_df = [c for c in onehot_cols if c in df.columns]
    if onehot_in_df:
        df = pd.get_dummies(df, columns=onehot_in_df, drop_first=True, dtype=float)

    for col, enc in encoders.items():
        if col == "__onehot__":
            continue
        if col in df.columns:
            df[col] = enc.transform(df[[col]])

    for col in feature_cols_post:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols_post]
    return df


def run_prediction(
    model_id: str,
    rows: list[dict],
    columns: list[str],
) -> dict[str, Any]:
    """Run predictions for ``rows`` using the model identified by ``model_id``."""
    log.info(">>> run_prediction  model=%s  rows=%d", model_id, len(rows))
    if not rows:
        return {"success": False, "error": "No rows provided for prediction."}

    try:
        bundle = load_model(model_id)
    except FileNotFoundError as exc:
        return {"success": False, "error": str(exc)}

    model = bundle["model"]
    pipeline = bundle.get("pipeline") or {}
    feature_cols_post: list[str] = list(bundle.get("feature_columns") or [])
    target_col: str = bundle.get("target_column") or ""
    task_type: str = bundle.get("task_type") or ""
    encoders = pipeline.get("encoders", {}) or {}

    raw_input = pd.DataFrame(rows, columns=columns)

    work = raw_input.copy()
    if target_col and target_col in work.columns:
        work = work.drop(columns=[target_col])

    raw_required = _infer_raw_required_columns(feature_cols_post, encoders)
    missing_required = [c for c in raw_required if c not in work.columns]
    if missing_required:
        return {
            "success": False,
            "error": (
                f"Missing required columns for prediction: {missing_required}. "
                f"Model expects raw columns: {raw_required}."
            ),
        }

    try:
        X_df = _apply_preprocessing(work, pipeline, feature_cols_post)
    except Exception as exc:
        log.exception("Preprocessing failed for model %s", model_id)
        return {"success": False, "error": f"Preprocessing failed: {exc}"}

    scaler = pipeline.get("scaler")
    try:
        X = scaler.transform(X_df.values) if scaler is not None else X_df.values
    except Exception as exc:
        log.exception("Scaler.transform failed for model %s", model_id)
        return {"success": False, "error": f"Scaling failed: {exc}"}

    try:
        preds = model.predict(X)
    except Exception as exc:
        log.exception("model.predict failed for model %s", model_id)
        return {"success": False, "error": f"Prediction failed: {exc}"}

    label_encoder = pipeline.get("label_encoder")
    if label_encoder is not None:
        try:
            preds_out = label_encoder.inverse_transform(np.asarray(preds).astype(int))
        except Exception:
            preds_out = preds
    else:
        preds_out = preds

    pred_col = f"predicted_{target_col}" if target_col else "prediction"
    out_df = raw_input.copy()
    out_df[pred_col] = preds_out

    if task_type == "classification" and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            out_df["probability"] = proba.max(axis=1).round(4)
        except Exception as exc:
            log.warning("predict_proba unavailable: %s", exc)

    out_records = out_df.where(pd.notnull(out_df), None).to_dict(orient="records")
    log.info("<<< run_prediction  predictions=%d", len(out_records))

    return {
        "success": True,
        "model_id": model_id,
        "task_type": task_type,
        "target_column": target_col,
        "prediction_column": pred_col,
        "rows": out_records,
        "columns": list(out_df.columns),
        "summary": {
            "n_predictions": len(out_df),
            "n_features_used": len(feature_cols_post),
            "n_input_columns": len(columns),
        },
    }

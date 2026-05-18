"""PredictAgent — load a saved model and produce predictions on new rows.

Reverses the preprocessing performed by train_agent._preprocess so the
incoming raw rows are transformed with the exact same scaler / encoders
that produced the training feature space.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.agents.feature_pipeline_storage import apply_pipeline, load_pipeline
from api.agents.model_storage import load_model
from api.agents.schema import infer_raw_required_columns
from api.logger import get_logger

log = get_logger(__name__)


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

    # Re-apply the same log-transform that training did on skewed features.
    # Stored in pipeline["log_features"] by _auto_feature_engineer; absent on models
    # trained before the FE step was added, so the get(..., []) keeps backward compat.
    log_features = pipeline.get("log_features", []) or []
    for col in log_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def run_prediction(
    model_id: str,
    rows: list[dict],
    columns: list[str],
    apply_feature_pipeline: bool | None = None,
) -> dict[str, Any]:
    """Run predictions for ``rows`` using the model identified by ``model_id``.

    When the model was trained on a /feature output, the model metadata stores
    a feature_pipeline_id. If the incoming columns look like RAW (matching the
    pipeline's pre-feature column set), this function replays /feature's
    transforms on the input before predicting — so the user can /predict on
    raw data even though the model was trained on engineered features.

    Set ``apply_feature_pipeline=True`` to force the replay, ``False`` to skip
    it, or leave ``None`` (default) to auto-detect by column overlap.
    """
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
    metadata = bundle.get("metadata") or {}
    feature_pipeline_id: str = metadata.get("feature_pipeline_id") or ""

    raw_input = pd.DataFrame(rows, columns=columns)

    # ── /feature pipeline replay ────────────────────────────────────────────
    # If the model was trained on engineered output and the user is now passing
    # RAW columns, run the saved feature pipeline on the input first.
    pipeline_applied = False
    pipeline_skip_reason = ""
    if feature_pipeline_id:
        state = load_pipeline(feature_pipeline_id)
        if state is not None:
            input_cols = set(raw_input.columns.astype(str))
            # Detection: when /feature uses label/ordinal/frequency encoding the
            # column NAMES don't change (only values do), so a set-based heuristic
            # can't distinguish raw from engineered. Look at dtypes instead: if
            # any encoder column is still non-numeric in the input, the input
            # hasn't been through /feature yet.
            needs_replay = False
            for col, info in state.encoders.items():
                if col in input_cols:
                    method = info.get("method")
                    if method in ("label", "ordinal", "frequency", "target"):
                        if not pd.api.types.is_numeric_dtype(raw_input[col]):
                            needs_replay = True
                            break
                elif info.get("method") == "onehot":
                    # Onehot expands col → col_value sub-columns. If those sub-
                    # columns are absent from input but the raw col IS present,
                    # it's raw.
                    if col in input_cols:
                        cats = (info.get("state") or {}).get("categories") or []
                        if cats and not all(c in input_cols for c in cats):
                            needs_replay = True
                            break
            # Also catch the no-encoder case: any raw col not present in engineered_columns
            extra_raw = [c for c in state.raw_columns if c in input_cols and c not in state.engineered_columns]
            if extra_raw:
                needs_replay = True

            if apply_feature_pipeline is True:
                should_apply = True
            elif apply_feature_pipeline is False:
                should_apply = False
            else:
                should_apply = needs_replay
            if should_apply:
                try:
                    raw_input = apply_pipeline(raw_input, state)
                    pipeline_applied = True
                    log.info(
                        "Applied feature pipeline %s → %d cols",
                        feature_pipeline_id, len(raw_input.columns),
                    )
                except Exception as exc:
                    pipeline_skip_reason = f"pipeline replay failed: {exc}"
                    log.warning(pipeline_skip_reason)
        else:
            pipeline_skip_reason = f"linked pipeline {feature_pipeline_id} not found"
            log.warning(pipeline_skip_reason)

    work = raw_input.copy()
    if target_col and target_col in work.columns:
        work = work.drop(columns=[target_col])

    raw_required = infer_raw_required_columns(feature_cols_post, encoders)
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

    # Reverse log1p applied to the target during training, if any.
    if task_type == "regression" and bool(pipeline.get("log_target", False)):
        preds = np.expm1(np.asarray(preds, dtype=float))

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
        "feature_pipeline_id": feature_pipeline_id,
        "feature_pipeline_applied": pipeline_applied,
        "feature_pipeline_skip_reason": pipeline_skip_reason,
    }

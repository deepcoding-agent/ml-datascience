"""Persistence + replay for /feature pipelines.

save_pipeline(state) → pipeline_id      writes a JSON file under PIPELINES_DIR
load_pipeline(pipeline_id) → state      reads it back
apply_pipeline(raw_df, state) → df      replays the transforms on new data

The state captured by feature_agent.run() is fully serializable (no fitted
sklearn objects — just the parameters they need). That keeps the storage
format simple and lets /predict re-apply transforms without re-importing
the original model file's pickled scaler/encoder.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from api.logger import get_logger
from api.models import FeaturePipelineState

log = get_logger(__name__)

# Stored alongside model artifacts. Same volume on Fly/Render.
PIPELINES_DIR = Path(
    os.environ.get("PIPELINES_DIR")
    or (os.environ.get("MODELS_DIR", "models") + "/pipelines")
).resolve()
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)
log.info("Feature pipeline storage dir: %s", PIPELINES_DIR)


# ── Persistence ──────────────────────────────────────────────────────────────


def save_pipeline(state: FeaturePipelineState, dataset_name: str = "") -> str:
    pipeline_id = state.pipeline_id or f"fp_{uuid.uuid4().hex[:12]}"
    state.pipeline_id = pipeline_id
    if dataset_name and not state.source_dataset_name:
        state.source_dataset_name = dataset_name
    if not state.created_at:
        state.created_at = datetime.now(timezone.utc).isoformat()

    path = PIPELINES_DIR / f"{pipeline_id}.json"
    path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    log.info("Saved feature pipeline %s (%s → %s)",
             pipeline_id, len(state.raw_columns), len(state.engineered_columns))
    return pipeline_id


def load_pipeline(pipeline_id: str) -> FeaturePipelineState | None:
    path = PIPELINES_DIR / f"{pipeline_id}.json"
    if not path.exists():
        return None
    try:
        return FeaturePipelineState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not parse pipeline %s: %s", pipeline_id, exc)
        return None


def pipeline_exists(pipeline_id: str) -> bool:
    return (PIPELINES_DIR / f"{pipeline_id}.json").exists()


# ── Replay ───────────────────────────────────────────────────────────────────


def _apply_derived(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    for spec in state.derived_specs:
        try:
            if spec.kind == "date_part" and len(spec.sources) == 1:
                src = spec.sources[0]
                if src not in df.columns:
                    continue
                col = pd.to_datetime(df[src], errors="coerce")
                part = spec.name.rsplit("_", 1)[-1]
                if part == "year":
                    df[spec.name] = col.dt.year
                elif part == "month":
                    df[spec.name] = col.dt.month
                elif part == "day":
                    df[spec.name] = col.dt.day
                elif part == "dayofweek":
                    df[spec.name] = col.dt.dayofweek
                elif part == "hour":
                    df[spec.name] = col.dt.hour
            elif spec.kind == "ratio" and len(spec.sources) == 2:
                a, b = spec.sources
                if a in df.columns and b in df.columns:
                    df[spec.name] = df[a] / df[b].replace({0: np.nan})
            elif spec.kind == "interaction" and len(spec.sources) == 2:
                a, b = spec.sources
                if a in df.columns and b in df.columns:
                    df[spec.name] = df[a] * df[b]
            elif spec.kind == "log" and len(spec.sources) == 1:
                src = spec.sources[0]
                if src in df.columns:
                    df[spec.name] = np.log1p(df[src].clip(lower=0))
        except Exception as exc:
            log.warning("apply derived %s failed: %s", spec.name, exc)
    return df


def _apply_null_fills(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    for col, info in state.null_fills.items():
        if col not in df.columns:
            continue
        strategy = info.get("strategy")
        if strategy in ("mean", "median", "mode", "constant"):
            df[col] = df[col].fillna(info.get("value"))
        elif strategy == "ffill":
            df[col] = df[col].ffill().bfill()
        # "drop" is intentionally skipped at predict time — we don't remove rows
    return df


def _apply_outliers(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    for col, bounds in state.outlier_bounds.items():
        if col not in df.columns:
            continue
        try:
            df[col] = df[col].clip(bounds["low"], bounds["high"])
        except Exception as exc:
            log.warning("apply outlier clip %s failed: %s", col, exc)
    return df


def _apply_encoders(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    for col, info in state.encoders.items():
        if col not in df.columns:
            continue
        method = info.get("method")
        st = info.get("state", {}) or {}
        try:
            series = df[col]
            if method == "onehot":
                categories: list[str] = st.get("categories", [])
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)
                dummies = dummies.reindex(columns=categories, fill_value=0)
                df = pd.concat([df.drop(columns=[col]), dummies.astype(int)], axis=1)
            elif method == "label":
                mapping = st.get("mapping", {})
                df[col] = series.astype(str).fillna("__nan__").map(mapping).fillna(-1).astype(int)
            elif method == "ordinal":
                cats: list[str] = st.get("categories", [])
                cat_to_int = {v: i for i, v in enumerate(cats)}
                df[col] = series.astype(str).fillna("__nan__").map(cat_to_int).fillna(-1).astype(int)
            elif method == "frequency":
                freq_map = st.get("freq_map", {})
                df[col] = series.astype(str).map(freq_map).fillna(0.0)
            elif method == "target":
                mean_map = st.get("mean_map", {})
                global_mean = st.get("global_mean", 0.0)
                df[col] = series.astype(str).map(mean_map).fillna(global_mean)
        except Exception as exc:
            log.warning("apply encode %s on %s failed: %s", method, col, exc)
    return df


def _apply_scaler(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    if not state.scaler:
        return df
    method = state.scaler.get("method")
    params: dict[str, dict] = state.scaler.get("params", {})
    for col, p in params.items():
        if col not in df.columns:
            continue
        try:
            if method == "standard":
                scale = p.get("scale", 1.0) or 1.0
                df[col] = (df[col] - p.get("mean", 0.0)) / scale
            elif method == "minmax":
                rng = (p.get("max", 1.0) - p.get("min", 0.0)) or 1.0
                df[col] = (df[col] - p.get("min", 0.0)) / rng
            elif method == "robust":
                scale = p.get("scale", 1.0) or 1.0
                df[col] = (df[col] - p.get("center", 0.0)) / scale
        except Exception as exc:
            log.warning("apply scale %s on %s failed: %s", method, col, exc)
    return df


def _apply_pca(df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    if not state.pca:
        return df
    cols: list[str] = state.pca.get("feature_cols", [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        log.warning("PCA replay missing columns: %s", missing)
        return df
    try:
        mean = np.asarray(state.pca.get("mean", []), dtype=float)
        components = np.asarray(state.pca.get("components", []), dtype=float)
        n_components = int(state.pca.get("n_components", components.shape[0]))
        target_col = state.target_column
        target_series = df[target_col] if target_col and target_col in df.columns else None
        X = df[cols].fillna(0).to_numpy(dtype=float)
        Xc = X - mean
        Z = Xc @ components.T                              # (n_rows, n_components)
        pcs = pd.DataFrame(
            Z, columns=[f"pc_{i+1}" for i in range(n_components)], index=df.index,
        )
        if target_series is not None:
            pcs[target_col] = target_series.values
        return pcs
    except Exception as exc:
        log.warning("PCA replay failed: %s", exc)
        return df


def apply_pipeline(raw_df: pd.DataFrame, state: FeaturePipelineState) -> pd.DataFrame:
    """Replay every step of /feature on `raw_df` using fitted state from `state`.

    The order matches feature_agent.run():
      1. drop columns          → drop
      2. derived features      → add new cols
      3. null fills            → fillna with saved values
      4. outlier bounds        → clip to saved bounds
      5. encoders              → label/ordinal/frequency/onehot/target using maps
      6. scaler                → standard/minmax/robust using saved params
      7. selection             → keep only columns kept at fit time
      8. PCA                   → project via saved components + mean

    Any column the pipeline doesn't know about is left alone — the caller
    (e.g. /predict) is responsible for re-checking the final schema.
    """
    df = raw_df.copy()

    # 1. drop
    drop_now = [c for c in state.dropped_columns if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    # 2. derived
    df = _apply_derived(df, state)

    # 3. nulls
    df = _apply_null_fills(df, state)

    # 4. outliers
    df = _apply_outliers(df, state)

    # 5. encoders
    df = _apply_encoders(df, state)

    # Drop any remaining non-numeric, non-datetime columns (text leftovers)
    target = state.target_column
    leftover = [
        c for c in df.columns
        if c != target
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if leftover:
        df = df.drop(columns=leftover)

    # 6. scaler
    df = _apply_scaler(df, state)

    # 7. selection — keep only the columns the model was actually trained on
    if state.kept_after_selection is not None:
        target = state.target_column
        keep = [c for c in state.kept_after_selection if c in df.columns]
        if target and target in df.columns and target not in keep:
            keep.append(target)
        df = df[keep] if keep else df

    # 8. PCA
    df = _apply_pca(df, state)

    return df

"""Feature engineering agent for /feature.

Two entry points:

    analyze(data, model_id)  → FeatureSuggestion
        Profile every column, propose a sensible FeatureConfig, list derived-
        feature candidates, infer task type + target. Used by the setup panel
        on mount to pre-fill all knobs.

    run(data, config, model_id)  → (engineered_df, FeatureReportArtifact,
                                    optional (train_df, test_df))
        Execute the pipeline the user confirmed in the panel. Returns one
        engineered dataset; if config.split == "train_test", also returns the
        split halves. Every step is recorded for the FeatureReportCard.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

from api.agents.context_analyzer import analyze_context
from api.llm import get_llm
from api.logger import get_logger
from api.models import (
    ColumnPlan,
    DerivedFeatureSpec,
    FeatureColumnProfile,
    FeatureConfig,
    FeatureDroppedItem,
    FeatureRankItem,
    FeatureReportArtifact,
    FeatureStep,
    FeatureSuggestion,
)

log = get_logger(__name__)


# ── ANALYSIS ─────────────────────────────────────────────────────────────────


def _column_role(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    sample = series.dropna().head(50)
    if sample.empty:
        return "categorical"
    avg_len = float(sample.astype(str).str.len().mean())
    if avg_len > 40 and series.nunique() > 30:
        return "text"
    return "categorical"


def _suggest_encoding(series: pd.Series) -> str | None:
    """Default-encoding heuristic that PRESERVES original column names:
    binary → label (0/1), low/mid cardinality → ordinal, high cardinality →
    frequency. Onehot is intentionally never the default — it produces
    column_value-suffix columns (mainroad_yes, mainroad_no) which most
    downstream tasks find harder to read than 0/1 in the original column.
    Users can still pick onehot per-column in the panel if they want it.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return None
    n_unique = int(series.nunique())
    if n_unique <= 1:
        return None
    if n_unique == 2:
        return "label"
    if n_unique <= 50:
        return "ordinal"
    return "frequency"


def _suggest_null_strategy(series: pd.Series) -> str:
    null_pct = float(series.isnull().mean() * 100)
    if null_pct == 0:
        return "none"
    if null_pct > 60:
        return "drop"
    if pd.api.types.is_numeric_dtype(series):
        return "median" if abs(float(series.skew(skipna=True) or 0)) > 1 else "mean"
    return "mode"


def _detect_task_and_targets(df: pd.DataFrame) -> tuple[str, list[str]]:
    """Heuristic: classification target → low-cardinality non-numeric or 2-20 unique numeric.
    Regression target → numeric with > 20 distinct values.
    Returns (task, ordered_target_candidates)."""
    candidates: list[tuple[str, str, int]] = []  # (col, kind, score)
    for col in df.columns:
        s = df[col]
        n_unique = int(s.nunique(dropna=True))
        if n_unique <= 1 or s.isnull().mean() > 0.8:
            continue
        if pd.api.types.is_numeric_dtype(s):
            if 2 <= n_unique <= 20:
                candidates.append((col, "classification", 10 + (20 - n_unique)))
            else:
                candidates.append((col, "regression", 5))
        else:
            if 2 <= n_unique <= 20:
                candidates.append((col, "classification", 12 + (20 - n_unique)))
    if not candidates:
        return "unsupervised", []
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_task = candidates[0][1]
    aligned = [c for c, k, _ in candidates if k == top_task]
    return top_task, aligned[:5]


def _propose_derived(df: pd.DataFrame, numeric_cols: list[str], datetime_cols: list[str]) -> list[DerivedFeatureSpec]:
    """Conservative derived-feature proposals — date parts + 1-2 safe ratios."""
    proposed: list[DerivedFeatureSpec] = []
    for col in datetime_cols[:3]:
        for part in ("year", "month", "dayofweek"):
            proposed.append(
                DerivedFeatureSpec(
                    name=f"{col}_{part}",
                    kind="date_part",
                    sources=[col],
                    detail=f"extract {part} from {col}",
                    enabled=True,
                )
            )
    if len(numeric_cols) >= 2:
        a, b = numeric_cols[0], numeric_cols[1]
        if (df[b].fillna(0) != 0).all():
            proposed.append(
                DerivedFeatureSpec(
                    name=f"{a}_per_{b}",
                    kind="ratio",
                    sources=[a, b],
                    detail=f"{a} / {b}",
                    enabled=False,
                )
            )
    return proposed[:6]


def _build_column_profiles(df: pd.DataFrame) -> list[FeatureColumnProfile]:
    profiles: list[FeatureColumnProfile] = []
    for col in df.columns:
        s = df[col]
        role = _column_role(s)
        null_pct = float(s.isnull().mean() * 100)
        n_unique = int(s.nunique(dropna=True))
        note = ""
        if n_unique <= 1:
            note = "constant — recommend drop"
        elif null_pct > 60:
            note = "high null %"
        elif role == "text":
            note = "free-form text — recommend drop"
        profiles.append(
            FeatureColumnProfile(
                name=col,
                dtype=str(s.dtype),
                role=role,
                null_pct=round(null_pct, 1),
                n_unique=n_unique,
                suggested_null_strategy=_suggest_null_strategy(s),
                suggested_encoding=_suggest_encoding(s),
                note=note,
            )
        )
    return profiles


ANALYZE_PROMPT = """\
You are a feature-engineering specialist. Given the dataset summary below, write ONE concise paragraph (3-5 sentences) explaining the recommended preprocessing strategy. Do NOT output JSON or markdown headers — just plain text.

DATASET
{summary}

PROFILES (top 12)
{profiles}

DETECTED TASK
{task} on target candidates: {targets}

Explain: (a) which task you'd pick, (b) what null/outlier strategy is suitable, (c) any columns to drop, (d) whether scaling and feature selection are useful here.
"""


def analyze(data: list[dict], model_id: str | None = None) -> FeatureSuggestion:
    df = pd.DataFrame(data)
    ctx = analyze_context(df)
    profiles = _build_column_profiles(df)
    detected_task, target_candidates = _detect_task_and_targets(df)

    # Build a default FeatureConfig from heuristics
    target = target_candidates[0] if target_candidates else None
    columns_plan: dict[str, ColumnPlan] = {}
    for p in profiles:
        if p.n_unique <= 1 or p.role == "text" or p.null_pct > 80:
            columns_plan[p.name] = ColumnPlan(drop=True)
            continue
        plan = ColumnPlan()
        if p.role == "categorical":
            plan.encoding = p.suggested_encoding
        if p.null_pct > 0:
            plan.null_strategy = p.suggested_null_strategy
        if plan != ColumnPlan():
            columns_plan[p.name] = plan

    # Outlier strategy: only suggest IQR clip if a meaningful fraction of numeric cols are skewed
    outlier_strategy = "iqr_clip" if len(ctx.skewed_cols) >= 2 else "none"

    # Scaling: standard by default; robust if many skewed cols
    scaling = "robust" if len(ctx.skewed_cols) >= 3 else "standard"

    # Feature selection: enable when many features
    selection_method = "mutual_info" if df.shape[1] > 15 else "none"
    selection_top_n = min(20, max(5, df.shape[1] - 1))

    derived = _propose_derived(df, ctx.numeric_cols, ctx.datetime_cols)

    suggested_config = FeatureConfig(
        task=detected_task,
        target_column=target,
        null_default="auto",
        outlier_strategy=outlier_strategy,
        scaling=scaling,
        encoding_default="auto",
        columns=columns_plan,
        selection_method=selection_method,
        selection_top_n=selection_top_n,
        derived=derived,
    )

    # LLM reasoning (optional polish — never blocks the suggestion)
    reasoning = ""
    try:
        llm = get_llm(temperature=0.0, max_tokens=300, model_id=model_id)
        summary = (
            f"shape={df.shape[0]}x{df.shape[1]}  duplicates={ctx.duplicate_count}  "
            f"numeric={len(ctx.numeric_cols)}  categorical={len(ctx.categorical_cols)}  "
            f"datetime={len(ctx.datetime_cols)}  skewed={len(ctx.skewed_cols)}"
        )
        profile_lines = [
            f"  {p.name} ({p.role}, null={p.null_pct}%, unique={p.n_unique})"
            for p in profiles[:12]
        ]
        prompt = ANALYZE_PROMPT.format(
            summary=summary,
            profiles="\n".join(profile_lines),
            task=detected_task,
            targets=", ".join(target_candidates) or "n/a",
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        reasoning = (resp.content or "").strip()
    except Exception as exc:  # pragma: no cover — never fail analyze
        log.warning("Feature analyze LLM polish failed: %s", exc)
        reasoning = (
            f"Detected {detected_task} task. {len(ctx.high_null_cols)} high-null columns "
            f"flagged for drop. {scaling.title()} scaling chosen. "
            f"Feature selection: {selection_method}."
        )

    return FeatureSuggestion(
        config=suggested_config,
        reasoning=reasoning,
        column_profiles=profiles,
        derived_candidates=derived,
        suggested_targets=target_candidates,
        detected_task=detected_task,
    )


# ── PIPELINE EXECUTION ───────────────────────────────────────────────────────


@dataclass
class _PipelineState:
    df: pd.DataFrame
    target: str | None
    task: str
    steps: list[FeatureStep]
    dropped: list[FeatureDroppedItem]
    derived_added: list[str]


def _resolve_null_strategy(plan_strategy: str | None, global_default: str, series: pd.Series) -> str:
    if plan_strategy and plan_strategy != "auto":
        return plan_strategy
    if global_default != "auto":
        return global_default
    return _suggest_null_strategy(series)


def _resolve_encoding(plan_encoding: str | None, global_default: str, series: pd.Series) -> str | None:
    if plan_encoding and plan_encoding != "auto":
        return None if plan_encoding == "none" else plan_encoding
    if global_default != "auto":
        return None if global_default == "none" else global_default
    return _suggest_encoding(series)


def _step_drop_columns(state: _PipelineState, config: FeatureConfig) -> None:
    to_drop: list[str] = []
    for name, plan in config.columns.items():
        if plan.drop and name in state.df.columns:
            to_drop.append(name)
            state.dropped.append(FeatureDroppedItem(name=name, reason="user/AI marked drop"))
    if to_drop:
        state.df = state.df.drop(columns=to_drop)
        state.steps.append(FeatureStep(kind="drop", detail=f"dropped {len(to_drop)} columns", affected=to_drop))


def _step_fill_nulls(state: _PipelineState, config: FeatureConfig) -> None:
    filled: dict[str, str] = {}
    for col in list(state.df.columns):
        if state.df[col].isnull().sum() == 0:
            continue
        plan = config.columns.get(col, ColumnPlan())
        strategy = _resolve_null_strategy(plan.null_strategy, config.null_default, state.df[col])
        if strategy == "none":
            continue
        if strategy == "drop":
            before = len(state.df)
            state.df = state.df.dropna(subset=[col])
            filled[col] = f"drop ({before - len(state.df)} rows)"
            continue
        try:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(state.df[col]):
                state.df[col] = state.df[col].fillna(state.df[col].mean())
                filled[col] = "mean"
            elif strategy == "median" and pd.api.types.is_numeric_dtype(state.df[col]):
                state.df[col] = state.df[col].fillna(state.df[col].median())
                filled[col] = "median"
            elif strategy == "mode":
                mode_vals = state.df[col].mode(dropna=True)
                if not mode_vals.empty:
                    state.df[col] = state.df[col].fillna(mode_vals.iloc[0])
                    filled[col] = "mode"
            elif strategy == "ffill":
                state.df[col] = state.df[col].ffill().bfill()
                filled[col] = "ffill"
            elif strategy == "constant" and plan.null_fill_value is not None:
                state.df[col] = state.df[col].fillna(plan.null_fill_value)
                filled[col] = f"constant ({plan.null_fill_value})"
        except Exception as exc:
            log.warning("null fill failed for %s (%s): %s", col, strategy, exc)
    if filled:
        state.steps.append(
            FeatureStep(
                kind="null_fill",
                detail=f"filled nulls in {len(filled)} columns",
                affected=list(filled.keys()),
                metrics=filled,
            )
        )


def _step_outliers(state: _PipelineState, config: FeatureConfig) -> None:
    strategy = config.outlier_strategy
    if strategy == "none":
        return
    threshold = config.outlier_threshold
    affected: dict[str, int] = {}
    numeric_cols = state.df.select_dtypes(include="number").columns.tolist()
    target = state.target
    for col in numeric_cols:
        if col == target:
            continue
        series = state.df[col]
        try:
            if strategy == "iqr_clip":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr <= 0:
                    continue
                low, high = q1 - threshold * iqr, q3 + threshold * iqr
                clipped = int(((series < low) | (series > high)).sum())
                if clipped:
                    state.df[col] = series.clip(low, high)
                    affected[col] = clipped
            elif strategy == "zscore_remove":
                mean, std = series.mean(), series.std()
                if std and std > 0:
                    mask = ((series - mean).abs() / std) <= threshold
                    removed = int((~mask).sum())
                    if removed:
                        state.df = state.df[mask].reset_index(drop=True)
                        affected[col] = removed
            elif strategy == "winsorize":
                low, high = series.quantile(0.05), series.quantile(0.95)
                clipped = int(((series < low) | (series > high)).sum())
                if clipped:
                    state.df[col] = series.clip(low, high)
                    affected[col] = clipped
        except Exception as exc:
            log.warning("outlier %s failed for %s: %s", strategy, col, exc)
    if affected:
        state.steps.append(
            FeatureStep(
                kind="outlier",
                detail=f"{strategy} on {len(affected)} columns",
                affected=list(affected.keys()),
                metrics=affected,
            )
        )


def _step_derived(state: _PipelineState, config: FeatureConfig) -> None:
    if not config.derived:
        return
    added: list[str] = []
    for spec in config.derived:
        if not spec.enabled:
            continue
        try:
            if spec.kind == "date_part" and len(spec.sources) == 1:
                src = spec.sources[0]
                if src not in state.df.columns:
                    continue
                col = pd.to_datetime(state.df[src], errors="coerce")
                part = spec.name.rsplit("_", 1)[-1]
                if part == "year":
                    state.df[spec.name] = col.dt.year
                elif part == "month":
                    state.df[spec.name] = col.dt.month
                elif part == "day":
                    state.df[spec.name] = col.dt.day
                elif part == "dayofweek":
                    state.df[spec.name] = col.dt.dayofweek
                elif part == "hour":
                    state.df[spec.name] = col.dt.hour
                added.append(spec.name)
            elif spec.kind == "ratio" and len(spec.sources) == 2:
                a, b = spec.sources
                if a in state.df.columns and b in state.df.columns:
                    denom = state.df[b].replace({0: np.nan})
                    state.df[spec.name] = state.df[a] / denom
                    added.append(spec.name)
            elif spec.kind == "interaction" and len(spec.sources) == 2:
                a, b = spec.sources
                if a in state.df.columns and b in state.df.columns:
                    state.df[spec.name] = state.df[a] * state.df[b]
                    added.append(spec.name)
            elif spec.kind == "log" and len(spec.sources) == 1:
                src = spec.sources[0]
                if src in state.df.columns:
                    state.df[spec.name] = np.log1p(state.df[src].clip(lower=0))
                    added.append(spec.name)
        except Exception as exc:
            log.warning("derived %s failed: %s", spec.name, exc)
    if added:
        state.derived_added.extend(added)
        state.steps.append(FeatureStep(kind="derive", detail=f"added {len(added)} derived columns", affected=added))


def _step_encode(state: _PipelineState, config: FeatureConfig) -> None:
    applied: dict[str, str] = {}
    target = state.target
    for col in list(state.df.columns):
        if col == target:
            continue
        series = state.df[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        plan = config.columns.get(col, ColumnPlan())
        method = _resolve_encoding(plan.encoding, config.encoding_default, series)
        if not method:
            continue
        try:
            if method == "onehot":
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)
                # cap explosion at 30 cols
                if dummies.shape[1] > 30:
                    method = "frequency"
                else:
                    state.df = pd.concat([state.df.drop(columns=[col]), dummies], axis=1)
                    applied[col] = "onehot"
                    continue
            if method == "label":
                le = LabelEncoder()
                state.df[col] = le.fit_transform(series.astype(str).fillna("__nan__"))
                applied[col] = "label"
            elif method == "ordinal":
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                state.df[col] = enc.fit_transform(series.astype(str).fillna("__nan__").to_frame()).flatten()
                applied[col] = "ordinal"
            elif method == "frequency":
                freqs = series.value_counts(normalize=True)
                state.df[col] = series.map(freqs).fillna(0)
                applied[col] = "frequency"
            elif method == "target" and target and target in state.df.columns:
                tgt = state.df[target]
                if pd.api.types.is_numeric_dtype(tgt):
                    means = state.df.groupby(col)[target].transform("mean")
                    state.df[col] = means
                    applied[col] = "target"
                else:
                    # fall back to label for non-numeric target
                    le = LabelEncoder()
                    state.df[col] = le.fit_transform(series.astype(str).fillna("__nan__"))
                    applied[col] = "label (fallback)"
        except Exception as exc:
            log.warning("encode %s failed for %s: %s", method, col, exc)
    if applied:
        state.steps.append(
            FeatureStep(
                kind="encode",
                detail=f"encoded {len(applied)} categorical columns",
                affected=list(applied.keys()),
                metrics=applied,
            )
        )

    # Drop any remaining non-numeric, non-datetime columns that weren't encoded
    # (free-form text etc.)
    leftover = [
        c
        for c in state.df.columns
        if c != target
        and not pd.api.types.is_numeric_dtype(state.df[c])
        and not pd.api.types.is_datetime64_any_dtype(state.df[c])
    ]
    if leftover:
        state.df = state.df.drop(columns=leftover)
        for c in leftover:
            state.dropped.append(FeatureDroppedItem(name=c, reason="non-numeric after encoding"))
        state.steps.append(
            FeatureStep(kind="drop", detail=f"dropped {len(leftover)} non-encodable columns", affected=leftover)
        )


def _step_scale(state: _PipelineState, config: FeatureConfig) -> None:
    if config.scaling == "none":
        return
    scaler_cls = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}.get(config.scaling)
    if not scaler_cls:
        return
    target = state.target
    numeric_cols = [c for c in state.df.select_dtypes(include="number").columns if c != target]
    if not numeric_cols:
        return
    try:
        scaler = scaler_cls()
        state.df[numeric_cols] = scaler.fit_transform(state.df[numeric_cols])
        state.steps.append(
            FeatureStep(kind="scale", detail=f"{config.scaling} scaling", affected=numeric_cols)
        )
    except Exception as exc:
        log.warning("scaling failed: %s", exc)


def _compute_rankings(
    df: pd.DataFrame, target: str | None, task: str, top_n: int
) -> list[FeatureRankItem]:
    if not target or target not in df.columns:
        # No target → variance ranking
        numeric = df.select_dtypes(include="number")
        var = numeric.var().sort_values(ascending=False).head(top_n)
        return [FeatureRankItem(name=str(c), score=float(v), method="variance") for c, v in var.items()]
    X = df.drop(columns=[target]).select_dtypes(include="number")
    y = df[target]
    if X.empty:
        return []
    try:
        if task == "regression":
            scores = mutual_info_regression(X.fillna(0), y, random_state=42)
        else:
            scores = mutual_info_classif(X.fillna(0), y, random_state=42)
        ranked = sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)[:top_n]
        return [FeatureRankItem(name=str(c), score=float(s), method="mutual_info") for c, s in ranked]
    except Exception as exc:
        log.warning("ranking failed: %s", exc)
        return []


def _step_select(state: _PipelineState, config: FeatureConfig) -> list[FeatureRankItem]:
    method = config.selection_method
    target = state.target
    top_n = max(1, config.selection_top_n)

    if method == "none":
        return _compute_rankings(state.df, target, state.task, top_n)

    # Build X/y for selectors that need them
    if not target or target not in state.df.columns:
        if method == "variance":
            try:
                numeric = state.df.select_dtypes(include="number")
                selector = VarianceThreshold(threshold=0.0)
                selector.fit(numeric.fillna(0))
                variances = pd.Series(selector.variances_, index=numeric.columns)
                ranked = variances.sort_values(ascending=False).head(top_n)
                keep = list(ranked.index)
                dropped = [c for c in numeric.columns if c not in keep]
                state.df = state.df.drop(columns=dropped)
                for c in dropped:
                    state.dropped.append(FeatureDroppedItem(name=c, reason="low variance"))
                state.steps.append(
                    FeatureStep(kind="select", detail=f"variance top {top_n}", affected=keep)
                )
                return [
                    FeatureRankItem(name=str(c), score=float(v), method="variance")
                    for c, v in ranked.items()
                ]
            except Exception as exc:
                log.warning("variance selection failed: %s", exc)
        return _compute_rankings(state.df, None, state.task, top_n)

    X = state.df.drop(columns=[target]).select_dtypes(include="number")
    y = state.df[target]
    if X.empty:
        return []

    rankings: list[FeatureRankItem] = []
    keep: list[str] = []
    try:
        if method == "mutual_info":
            if state.task == "regression":
                scores = mutual_info_regression(X.fillna(0), y, random_state=42)
            else:
                scores = mutual_info_classif(X.fillna(0), y, random_state=42)
            ranked = sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)[:top_n]
            keep = [c for c, _ in ranked]
            rankings = [FeatureRankItem(name=str(c), score=float(s), method="mutual_info") for c, s in ranked]
        elif method == "correlation":
            if state.task == "regression":
                corr = X.fillna(0).corrwith(y).abs().sort_values(ascending=False)
            else:
                # encode target for corr
                y_enc = LabelEncoder().fit_transform(y.astype(str))
                corr = X.fillna(0).apply(lambda c: abs(np.corrcoef(c, y_enc)[0, 1] or 0))
                corr = corr.sort_values(ascending=False)
            ranked = corr.head(top_n)
            keep = list(ranked.index)
            rankings = [FeatureRankItem(name=str(c), score=float(s), method="correlation") for c, s in ranked.items()]
        elif method == "variance":
            selector = VarianceThreshold(threshold=0.0)
            selector.fit(X.fillna(0))
            variances = pd.Series(selector.variances_, index=X.columns)
            ranked = variances.sort_values(ascending=False).head(top_n)
            keep = list(ranked.index)
            rankings = [FeatureRankItem(name=str(c), score=float(v), method="variance") for c, v in ranked.items()]
        elif method == "rfe":
            estimator = Ridge() if state.task == "regression" else LogisticRegression(max_iter=200)
            rfe = RFE(estimator=estimator, n_features_to_select=min(top_n, X.shape[1]))
            rfe.fit(X.fillna(0), y)
            ranking_scores = -rfe.ranking_  # lower rank value = better; flip so higher=better
            ranked = sorted(zip(X.columns, ranking_scores), key=lambda x: x[1], reverse=True)[:top_n]
            keep = [c for c, _ in ranked]
            rankings = [FeatureRankItem(name=str(c), score=float(s), method="rfe") for c, s in ranked]
    except Exception as exc:
        log.warning("feature selection %s failed: %s", method, exc)
        return _compute_rankings(state.df, target, state.task, top_n)

    if keep:
        dropped = [c for c in X.columns if c not in keep]
        state.df = state.df.drop(columns=dropped)
        for c in dropped:
            state.dropped.append(FeatureDroppedItem(name=c, reason=f"not in {method} top {top_n}"))
        state.steps.append(
            FeatureStep(
                kind="select",
                detail=f"{method} → kept top {len(keep)}",
                affected=keep,
            )
        )
    return rankings


def _step_reduce(state: _PipelineState, config: FeatureConfig) -> None:
    if config.reduction_method != "pca":
        return
    target = state.target
    numeric_cols = [c for c in state.df.select_dtypes(include="number").columns if c != target]
    if len(numeric_cols) <= config.reduction_n_components:
        return
    try:
        pca = PCA(n_components=config.reduction_n_components, random_state=42)
        components = pca.fit_transform(state.df[numeric_cols].fillna(0))
        new_cols = [f"pc_{i+1}" for i in range(config.reduction_n_components)]
        reduced = pd.DataFrame(components, columns=new_cols, index=state.df.index)
        target_series = state.df[target] if target and target in state.df.columns else None
        state.df = reduced
        if target_series is not None:
            state.df[target] = target_series.values
        state.steps.append(
            FeatureStep(
                kind="reduce",
                detail=f"PCA → {config.reduction_n_components} components",
                affected=new_cols,
                metrics={"explained_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_]},
            )
        )
    except Exception as exc:
        log.warning("PCA failed: %s", exc)


def run(
    data: list[dict],
    config: FeatureConfig,
) -> tuple[pd.DataFrame, FeatureReportArtifact, pd.DataFrame | None, pd.DataFrame | None]:
    """Execute the feature pipeline. Returns (engineered_df, report, train_df?, test_df?)."""
    started = time.perf_counter()
    df = pd.DataFrame(data)
    rows_before, cols_before = df.shape
    target = config.target_column if config.target_column and config.target_column in df.columns else None

    state = _PipelineState(
        df=df,
        target=target,
        task=config.task,
        steps=[],
        dropped=[],
        derived_added=[],
    )

    try:
        _step_drop_columns(state, config)
        _step_derived(state, config)
        _step_fill_nulls(state, config)
        _step_outliers(state, config)
        _step_encode(state, config)
        _step_scale(state, config)
        rankings = _step_select(state, config)
        _step_reduce(state, config)
    except Exception as exc:
        log.exception("feature pipeline failed: %s", exc)
        rankings = []
        report = FeatureReportArtifact(
            success=False,
            task=config.task,
            target_column=target,
            rows_before=rows_before,
            cols_before=cols_before,
            rows_after=state.df.shape[0],
            cols_after=state.df.shape[1],
            steps=state.steps,
            rankings=rankings,
            dropped_columns=state.dropped,
            derived_columns=state.derived_added,
            duration_seconds=round(time.perf_counter() - started, 3),
            error=str(exc),
        )
        return state.df, report, None, None

    train_df = test_df = None
    if config.split == "train_test" and target and target in state.df.columns:
        try:
            stratify = state.df[target] if config.task == "classification" else None
            train_df, test_df = train_test_split(
                state.df,
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=stratify,
            )
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            state.steps.append(
                FeatureStep(
                    kind="split",
                    detail=f"train_test split test_size={config.test_size}",
                    affected=[],
                    metrics={"train_rows": len(train_df), "test_rows": len(test_df)},
                )
            )
        except Exception as exc:
            log.warning("split failed: %s", exc)
            train_df = test_df = None

    rows_after, cols_after = state.df.shape
    report = FeatureReportArtifact(
        success=True,
        task=config.task,
        target_column=target,
        rows_before=rows_before,
        cols_before=cols_before,
        rows_after=rows_after,
        cols_after=cols_after,
        steps=state.steps,
        rankings=rankings,
        dropped_columns=state.dropped,
        derived_columns=state.derived_added,
        duration_seconds=round(time.perf_counter() - started, 3),
    )
    return state.df, report, train_df, test_df

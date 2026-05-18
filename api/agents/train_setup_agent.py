"""Analyzer for the /train setup panel.

`analyze_train(data, model_id) → TrainSuggestion`

Heuristic seeds (task type, target candidate, sample-size-aware CV/Optuna
sizing) feed a focused LLM call that can override any field after reading
the actual column names. Falls back to the heuristic cleanly when the
LLM is unavailable or returns invalid JSON.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage

from api.agents.feature_agent import _build_column_profiles, _detect_task_and_targets
from api.llm import get_llm
from api.logger import get_logger
from api.models import TrainSuggestion

log = get_logger(__name__)


_ALLOWED_TASK = {"auto", "classification", "regression", "clustering"}
_ALLOWED_METRIC = {
    "auto", "accuracy", "f1", "auc", "precision", "recall",
    "r2", "rmse", "mae", "silhouette",
}


def _heuristic_train_setup(df: pd.DataFrame) -> TrainSuggestion:
    """Pure heuristic — used as seed + fallback."""
    n_rows, n_cols = df.shape
    detected_task, candidates = _detect_task_and_targets(df)
    target = candidates[0] if candidates else None

    # Dataset-size-aware sizing
    if n_rows < 200:
        cv_folds = 3                    # less per-fold variance noise
        tun_trials = 20                 # save time
        test_size = 0.15                # keep more train data
    elif n_rows < 2000:
        cv_folds = 5
        tun_trials = 50
        test_size = 0.2
    else:
        cv_folds = 5
        tun_trials = 30                 # each trial is expensive on big data
        test_size = 0.2

    # Default scoring per task — auto is OK but we pick more specific defaults
    # when the heuristic is confident about the task.
    if detected_task == "classification":
        # Check class balance for imbalanced → f1 over accuracy
        scoring_metric = "auto"
        if target and target in df.columns:
            freqs = df[target].value_counts(normalize=True)
            if not freqs.empty and freqs.max() / max(freqs.min(), 1e-9) > 5:
                scoring_metric = "f1"   # imbalanced → f1
    elif detected_task == "regression":
        scoring_metric = "r2"
    elif detected_task == "clustering":
        scoring_metric = "silhouette"
    else:
        scoring_metric = "auto"

    return TrainSuggestion(
        target_column=target,
        task_type=detected_task if detected_task in _ALLOWED_TASK else "auto",
        scoring_metric=scoring_metric,
        cv_folds=cv_folds,
        tun_trials=tun_trials,
        test_size=test_size,
        suggested_targets=candidates[:5],
    )


LLM_PROMPT = """\
You are an AutoML configuration specialist. Pick the best training setup for the dataset below. Use the column names and stats — different domains need different setups.

DATASET
shape={n_rows}x{n_cols}  numeric={n_numeric}  categorical={n_categorical}  datetime={n_datetime}

COLUMNS
{column_lines}

HEURISTIC STARTING POINT (override fields you disagree with)
{heuristic_json}

CONFIG OPTIONS
- task_type      : auto | classification | regression | clustering
- scoring_metric : auto | accuracy | f1 | auc | precision | recall | r2 | rmse | mae | silhouette
- cv_folds       : 2-10  (small data → fewer folds)
- tun_trials     : 10-200 (more trials = slower; big data → fewer)
- test_size      : 0.05-0.4

GUIDELINES
- Pick target_column by reading column NAMES, not just cardinality. Common names: "churn", "label", "target", "outcome", "fraud", "default", "y", "class", "diagnosis", "survived", "price", "salary", "rating".
- ID-like columns (customer_id, uuid, user_id, transaction_id) are NEVER targets.
- For imbalanced binary classification (rare class < 15%), prefer scoring_metric="f1" or "auc" over accuracy.
- For regression with skewed/outlier-prone targets, prefer "rmse" over "r2".
- For clustering, target must be null.
- task_type="auto" only when you're genuinely unsure — prefer to commit.

Output ONLY valid JSON, no markdown, no commentary:
{{
  "target_column": "..." or null,
  "task_type": "...",
  "scoring_metric": "...",
  "cv_folds": 5,
  "tun_trials": 50,
  "test_size": 0.2,
  "reasoning": "ONE paragraph (3-4 sentences) explaining your picks given this dataset"
}}
"""


def _parse_llm_train(raw: str, base: TrainSuggestion, valid_cols: set[str]) -> TrainSuggestion | None:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    out = base.model_copy()

    tc = parsed.get("target_column")
    if tc is None:
        out.target_column = None
    elif isinstance(tc, str) and tc in valid_cols:
        out.target_column = tc

    if isinstance(parsed.get("task_type"), str) and parsed["task_type"] in _ALLOWED_TASK:
        out.task_type = parsed["task_type"]
    if isinstance(parsed.get("scoring_metric"), str) and parsed["scoring_metric"] in _ALLOWED_METRIC:
        out.scoring_metric = parsed["scoring_metric"]

    if isinstance(parsed.get("cv_folds"), int):
        out.cv_folds = max(2, min(10, parsed["cv_folds"]))
    if isinstance(parsed.get("tun_trials"), int):
        out.tun_trials = max(10, min(200, parsed["tun_trials"]))
    val = parsed.get("test_size")
    if isinstance(val, (int, float)):
        out.test_size = max(0.05, min(0.4, float(val)))

    reasoning = parsed.get("reasoning")
    if isinstance(reasoning, str):
        out.reasoning = reasoning.strip()
    return out


def analyze_train(data: list[dict], model_id: str | None = None) -> TrainSuggestion:
    df = pd.DataFrame(data)
    base = _heuristic_train_setup(df)

    valid_cols = set(df.columns.astype(str))
    profiles = _build_column_profiles(df)
    n_numeric = sum(1 for p in profiles if p.role == "numeric")
    n_categorical = sum(1 for p in profiles if p.role == "categorical")
    n_datetime = sum(1 for p in profiles if p.role == "datetime")

    try:
        llm = get_llm(temperature=0.0, max_tokens=500, model_id=model_id)
        column_lines = [
            f"  {p.name:24s}  role={p.role:11s}  unique={p.n_unique:5d}  null={p.null_pct:5.1f}%"
            for p in profiles[:30]
        ]
        if len(profiles) > 30:
            column_lines.append(f"  … +{len(profiles) - 30} more columns")
        heuristic_json = json.dumps(
            {
                "target_column": base.target_column,
                "task_type": base.task_type,
                "scoring_metric": base.scoring_metric,
                "cv_folds": base.cv_folds,
                "tun_trials": base.tun_trials,
                "test_size": base.test_size,
            },
            ensure_ascii=False,
            indent=2,
        )
        prompt = LLM_PROMPT.format(
            n_rows=df.shape[0],
            n_cols=df.shape[1],
            n_numeric=n_numeric,
            n_categorical=n_categorical,
            n_datetime=n_datetime,
            column_lines="\n".join(column_lines),
            heuristic_json=heuristic_json,
        )
        resp = llm.invoke([HumanMessage(content=prompt)])
        parsed = _parse_llm_train((resp.content or ""), base, valid_cols)
        if parsed is not None:
            overrides = {
                k: getattr(parsed, k)
                for k in ("target_column", "task_type", "scoring_metric", "cv_folds", "tun_trials", "test_size")
                if getattr(parsed, k) != getattr(base, k)
            }
            log.info("Train analyze LLM accepted — overrides: %s", overrides)
            return parsed
        log.warning("Train analyze LLM returned unparseable JSON — using heuristic")
    except Exception as exc:  # pragma: no cover — never block
        log.warning("Train analyze LLM call failed: %s — using heuristic", exc)

    # Fallback reasoning — must always be set, even without LLM
    if not base.reasoning:
        base.reasoning = (
            f"{base.task_type.title() if base.task_type != 'auto' else 'Auto-detected'} task; "
            f"suggested target '{base.target_column or '(none)'}'. "
            f"Sized for {df.shape[0]:,} rows × {df.shape[1]} cols: "
            f"{base.cv_folds}-fold CV, {base.tun_trials} Optuna trials, "
            f"test size {int(base.test_size * 100)}%. Scoring metric '{base.scoring_metric}'."
        )
    return base

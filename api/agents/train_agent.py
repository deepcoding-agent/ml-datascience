"""Train Agent — AI Auto ML Training Pipeline.

Orchestrates the full training pipeline:
  1. Analyze dataset → detect task type
  2. Auto-preprocess if needed (nulls, encoding, scaling)
  3. Select algorithms from ModelRegistry
  4. Cross-validate all candidates
  5. Hyperparameter tuning on top 3
  6. Final evaluation with charts
  7. Save model + return results
"""
from __future__ import annotations

import gc
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from api.agents.model_evaluator import (
    build_charts,
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_regression_metrics,
    get_classification_report_text,
)
from api.agents.model_registry import (
    get_algorithms_for_task,
    instantiate_model,
    select_algorithms,
)
from api.agents.model_storage import save_model
from api.llm import get_llm
from api.logger import get_logger

log = get_logger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress joblib/loky resource tracker warnings at shutdown
import os as _os
_os.environ.setdefault("LOKY_PICKLER", "pickle")
try:
    import joblib.externals.loky.backend.resource_tracker as _rt
    _rt.warnings.filterwarnings("ignore", module="resource_tracker")
except Exception:
    pass


# ── Step 1: Analyze ──────────────────────────────────────────────────────────

def _detect_task_type(
    y: pd.Series | None,
    user_task_type: str = "auto",
) -> str:
    """Detect classification, regression, or clustering.

    Key insight: JSON deserialization often turns numeric columns into strings.
    Always try numeric conversion before deciding based on dtype.
    """
    if user_task_type != "auto":
        return user_task_type
    if y is None:
        return "clustering"

    # Try to coerce object/string columns to numeric — JSON often stores numbers as strings
    if y.dtype == "object":
        coerced = pd.to_numeric(y, errors="coerce")
        if coerced.notna().sum() / max(len(y), 1) > 0.8:
            # Mostly numeric values stored as strings — treat as numeric
            y = coerced

    if y.dtype == "bool":
        return "classification"

    n_unique = y.nunique()
    n_rows = max(len(y), 1)

    # Many unique values relative to dataset size → regression
    # e.g. price column with 200 unique values out of 545 rows
    if n_unique > 20:
        return "regression"

    # Few unique values AND low cardinality ratio → classification
    if n_unique <= 20 and n_unique / n_rows < 0.05:
        return "classification"

    # Remaining object/category columns with ≤20 unique → classification
    if y.dtype in ("object", "category"):
        return "classification"

    return "regression"


# ── Step 2: Auto-preprocess ──────────────────────────────────────────────────

def _auto_preprocess(
    df: pd.DataFrame,
    target_column: str | None,
    task_type: str,
    test_size: float = 0.2,
) -> dict:
    """Smart preprocessing for training — proper encoding for real performance.

    Encoding strategy:
      - Low cardinality categoricals (< 15 unique): one-hot encoding
      - High cardinality categoricals (>= 15 unique): ordinal encoding
      - Numeric string columns: coerce to float
      - ID-like columns (sequential ints, all unique): drop
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column], errors="ignore") if target_column else df.copy()
    y = df[target_column].copy() if target_column and target_column in df.columns else None

    # ── Coerce string-encoded numbers (JSON artifact) ──
    if y is not None and y.dtype == "object":
        coerced = pd.to_numeric(y, errors="coerce")
        if coerced.notna().sum() / max(len(y), 1) > 0.8:
            y = coerced
            y = y.fillna(y.median())

    for col in X.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(X[col], errors="coerce")
        if coerced.notna().sum() / max(len(X), 1) > 0.8:
            X[col] = coerced

    # ── Drop ID-like columns (all unique ints, or named *id) ──
    drop_ids = []
    for col in X.columns:
        if X[col].dtype in ("int64", "float64"):
            if X[col].nunique() == len(X) and X[col].is_monotonic_increasing:
                drop_ids.append(col)
        if col.lower().strip() in ("id", "index", "row_id", "row_number"):
            drop_ids.append(col)
    if drop_ids:
        drop_ids = list(set(drop_ids))
        log.info("  Dropping ID-like columns: %s", drop_ids)
        X = X.drop(columns=drop_ids, errors="ignore")

    # ── Drop all-null columns ──
    null_cols = X.columns[X.isnull().all()].tolist()
    if null_cols:
        X = X.drop(columns=null_cols)

    # ── Fill nulls ──
    for col in X.select_dtypes(include="number").columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")

    # ── Encode categoricals ──
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: dict = {}
    onehot_cols: list[str] = []
    ordinal_cols: list[str] = []

    for col in cat_cols:
        n_unique = X[col].nunique()
        if n_unique < 15:
            onehot_cols.append(col)
        else:
            ordinal_cols.append(col)

    # One-hot encode low-cardinality categoricals (proper ML encoding)
    if onehot_cols:
        X = pd.get_dummies(X, columns=onehot_cols, drop_first=True, dtype=float)
        encoders["__onehot__"] = onehot_cols

    # Ordinal encode high-cardinality categoricals (fallback)
    for col in ordinal_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[col] = enc.fit_transform(X[[col]])
        encoders[col] = enc

    # ── Encode target for classification ──
    label_encoder = None
    class_names: list[str] = []
    if y is not None and task_type == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
        class_names = [str(c) for c in label_encoder.classes_]
    elif y is not None:
        y = y.astype(float)

    # ── Log-transform skewed regression targets ──
    # Right-skewed targets (housing prices, salaries, counts) hurt RMSE because
    # outliers dominate. log1p compresses the tail; expm1 reverses for metrics.
    log_target = False
    if y is not None and task_type == "regression":
        try:
            target_skew = float(y.skew())
            if abs(target_skew) > 1.0 and bool((y > -1.0).all()):
                y = pd.Series(np.log1p(y.values), name=y.name)
                log_target = True
                log.info("  Log-transformed target (skew=%.2f)", target_skew)
        except Exception as exc:
            log.warning("  Skew check failed for target: %s", exc)

    feature_names = X.columns.tolist()

    # ── Scale numeric features ──
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)

    # ── Train/test split ──
    if y is not None:
        stratify = y if task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=stratify,
            )
        except ValueError:
            log.warning("Stratified split failed — falling back to random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
            )
    else:
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = None, None

    log.info("  Encoding: one-hot=%s  ordinal=%s  features=%d",
             onehot_cols, ordinal_cols, len(feature_names))

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "encoders": encoders,
        "log_target": log_target,
    }


# ── Step 4: Cross-validate ───────────────────────────────────────────────────

def _safe_cv_folds(y_train: np.ndarray | None, requested_folds: int, task_type: str) -> int:
    """Reduce CV folds so every class has >= n_splits members."""
    if task_type != "classification" or y_train is None:
        return min(requested_folds, max(2, len(y_train) if y_train is not None else 2))
    _, counts = np.unique(y_train, return_counts=True)
    min_class = int(counts.min())
    safe = max(2, min(requested_folds, min_class))
    if safe < requested_folds:
        log.info("  Reduced CV folds %d→%d (smallest class has %d members)", requested_folds, safe, min_class)
    return safe


def _cross_validate_all(
    algo_keys: list[str],
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> list[dict]:
    """Run cross-validation for all selected algorithms.

    Auto-reduces folds for small datasets and falls back from
    StratifiedKFold → KFold → simple fit if CV is impossible.
    """
    results: list[dict] = []
    registry = get_algorithms_for_task(task_type)
    actual_folds = _safe_cv_folds(y_train, cv_folds, task_type)

    for algo_key in algo_keys:
        entry = registry[algo_key]
        model = instantiate_model(task_type, algo_key)
        t0 = time.time()

        try:
            if task_type == "clustering":
                model.fit(X_train)
                labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
                metrics = compute_clustering_metrics(np.array(X_train), np.array(labels))
                duration = time.time() - t0
                results.append({
                    "algo_key": algo_key,
                    "display_name": entry["display_name"],
                    "cv_metrics": metrics,
                    "train_time": duration,
                    "model": model,
                })
                continue

            if task_type == "classification":
                scoring = {"accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted", "recall": "recall_weighted"}
            else:
                scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}

            # Try StratifiedKFold → KFold → simple fit
            cv_results = None
            for cv_strategy in ("stratified", "kfold", "none"):
                try:
                    if cv_strategy == "stratified" and task_type == "classification":
                        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
                    elif cv_strategy == "kfold":
                        folds = min(actual_folds, len(X_train))
                        if folds < 2:
                            continue
                        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
                    elif cv_strategy == "none":
                        # Last resort: simple fit, evaluate on train set
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_train)
                        if task_type == "classification":
                            from sklearn.metrics import accuracy_score, f1_score
                            cv_metrics_dict = {
                                "accuracy_mean": float(accuracy_score(y_train, y_pred)),
                                "accuracy_std": 0.0,
                                "f1_mean": float(f1_score(y_train, y_pred, average="weighted", zero_division=0)),
                                "f1_std": 0.0,
                            }
                        else:
                            from sklearn.metrics import mean_squared_error, r2_score
                            cv_metrics_dict = {
                                "rmse_mean": float(np.sqrt(mean_squared_error(y_train, y_pred))),
                                "rmse_std": 0.0,
                                "r2_mean": float(r2_score(y_train, y_pred)),
                                "r2_std": 0.0,
                            }
                        duration = time.time() - t0
                        results.append({
                            "algo_key": algo_key,
                            "display_name": entry["display_name"],
                            "cv_metrics": cv_metrics_dict,
                            "train_time": duration,
                            "model": model,
                        })
                        log.info("  Fit done (no CV): %s  (%.1fs)", entry["display_name"], duration)
                        cv_results = True  # sentinel — already appended
                        break
                    else:
                        continue

                    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=2, return_train_score=False)
                    break  # success
                except ValueError:
                    continue

            if cv_results is None:
                log.warning("  All CV strategies failed for %s", entry["display_name"])
                continue
            if cv_results is True:
                continue  # already appended in "none" branch

            duration = time.time() - t0
            cv_metrics: dict[str, float] = {}
            for metric_name in scoring:
                vals = cv_results[f"test_{metric_name}"]
                if metric_name in ("rmse", "mae"):
                    vals = -vals
                cv_metrics[f"{metric_name}_mean"] = float(np.mean(vals))
                cv_metrics[f"{metric_name}_std"] = float(np.std(vals))

            results.append({
                "algo_key": algo_key,
                "display_name": entry["display_name"],
                "cv_metrics": cv_metrics,
                "train_time": duration,
                "model": model,
            })
            log.info("  CV done: %s  (%.1fs)", entry["display_name"], duration)

        except Exception as exc:
            log.warning("  CV failed for %s: %s", entry["display_name"], exc)
            continue

    return results


# ── Step 5: Hyperparameter tuning ────────────────────────────────────────────

def _baseline_cv_score(
    model: Any,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray | None,
    cv_folds: int,
) -> float:
    """CV score on the same metric Optuna would optimize — used when an algorithm
    has no tunable params so it can compete fairly with tuned models."""
    if task_type == "clustering":
        try:
            model.fit(X_train)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if n_labels < 2:
                return -1.0
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(X_train, labels))
        except Exception:
            return -1.0

    scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
    safe_folds = _safe_cv_folds(y_train, cv_folds, task_type)
    from sklearn.model_selection import cross_val_score
    for cv_strat in ("stratified", "kfold", "fit"):
        try:
            if cv_strat == "stratified" and task_type == "classification":
                cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
            elif cv_strat == "kfold":
                cv = KFold(n_splits=max(2, min(safe_folds, len(X_train))), shuffle=True, random_state=42)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_train)
                if task_type == "classification":
                    from sklearn.metrics import f1_score
                    return float(f1_score(y_train, y_pred, average="weighted", zero_division=0))
                return float(-np.sqrt(np.mean((y_train - y_pred) ** 2)))
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=2)
            return float(np.mean(scores))
        except ValueError:
            continue
    return float("-inf")


def _tune_model(
    algo_key: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> dict:
    """Tune hyperparameters with Optuna for a single algorithm."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    registry = get_algorithms_for_task(task_type)
    entry = registry[algo_key]
    tunable = entry.get("tunable_params", {})

    if not tunable:
        # No hyperparameters to search — still need a real CV score on the
        # same metric scale as tuned algos so model selection compares fairly.
        # (Was returning 0.0, which beat any negative neg_RMSE for regression.)
        model = instantiate_model(task_type, algo_key)
        baseline_score = _baseline_cv_score(model, task_type, X_train, y_train, cv_folds)
        if y_train is not None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)
        return {"model": model, "best_params": {}, "best_score": baseline_score}

    def objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {}
        for pname, spec in tunable.items():
            ptype = spec[0]
            if ptype == "int":
                params[pname] = trial.suggest_int(pname, spec[1], spec[2])
            elif ptype == "float":
                params[pname] = trial.suggest_float(pname, spec[1], spec[2])
            elif ptype == "float_log":
                params[pname] = trial.suggest_float(pname, spec[1], spec[2], log=True)
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, spec[1])

        model = instantiate_model(task_type, algo_key, params)

        if task_type == "clustering":
            model.fit(X_train)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if n_labels < 2:
                return -1.0
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(X_train, labels))

        safe_folds = _safe_cv_folds(y_train, cv_folds, task_type)
        if task_type == "classification":
            scoring = "f1_weighted"
        else:
            scoring = "neg_root_mean_squared_error"

        from sklearn.model_selection import cross_val_score
        # Try stratified → kfold → simple fit
        for cv_strat in ("stratified", "kfold", "fit"):
            try:
                if cv_strat == "stratified" and task_type == "classification":
                    cv = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=42)
                elif cv_strat == "kfold":
                    cv = KFold(n_splits=max(2, min(safe_folds, len(X_train))), shuffle=True, random_state=42)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_train)
                    if task_type == "classification":
                        from sklearn.metrics import f1_score
                        return float(f1_score(y_train, y_pred, average="weighted", zero_division=0))
                    else:
                        return float(-np.sqrt(np.mean((y_train - y_pred) ** 2)))
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=2)
                return float(np.mean(scores))
            except ValueError:
                continue
        return -1.0

    # Baseline (default-params) CV score — safe fallback when tuning makes it worse,
    # and ensures we never return a tuned model that's worse than vanilla defaults.
    baseline_model = instantiate_model(task_type, algo_key)
    baseline_score = _baseline_cv_score(baseline_model, task_type, X_train, y_train, cv_folds)

    # Adaptive trial count — tuned for speed on small datasets and depth on large.
    # 8 trials per tunable dimension is the search-space rule of thumb; we then
    # CAP based on dataset size so tiny datasets don't pay for huge Optuna runs
    # that won't generalize better than a modest search would.
    n_tunable = len(tunable)
    adaptive_trials = max(n_trials, n_tunable * 8)
    n_rows = len(X_train)
    if n_rows < 500:
        adaptive_trials = min(adaptive_trials, max(n_trials, 15))   # tiny → fast
    elif n_rows < 2000:
        adaptive_trials = min(adaptive_trials, max(n_trials, 30))   # small → moderate
    else:
        adaptive_trials = min(adaptive_trials, 80)                  # large → full search
    log.info("    %s tuning: trials=%d (requested=%d, tunable=%d, rows=%d) baseline=%.4f",
             entry["display_name"], adaptive_trials, n_trials, n_tunable, len(X_train), baseline_score)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=adaptive_trials,
        callbacks=[lambda _study, _trial: gc.collect()],
        show_progress_bar=False,
    )

    tuned_score = float(study.best_value)

    # Safe fallback: if tuning failed to beat default params, keep defaults.
    # This guarantees tuning can never make a model worse than out-of-the-box.
    if tuned_score < baseline_score:
        log.info("    %s fallback: tuned=%.4f < baseline=%.4f → keeping default params",
                 entry["display_name"], tuned_score, baseline_score)
        if y_train is not None:
            baseline_model.fit(X_train, y_train)
        else:
            baseline_model.fit(X_train)
        return {"model": baseline_model, "best_params": {}, "best_score": baseline_score}

    best_params = study.best_params
    best_model = instantiate_model(task_type, algo_key, best_params)
    if y_train is not None:
        best_model.fit(X_train, y_train)
    else:
        best_model.fit(X_train)

    return {"model": best_model, "best_params": best_params, "best_score": tuned_score}


# ── Step 7: AI Summary ──────────────────────────────────────────────────────

def _generate_ai_summary(
    task_type: str,
    best_algo: str,
    metrics: dict,
    comparison: list[dict],
    feature_importance: list[dict],
    model_id: str | None = None,
) -> str:
    """Generate a brief AI-powered summary of the training results."""
    try:
        llm = get_llm(temperature=0.3, max_tokens=1024, model_id=model_id)

        context = f"""Task: {task_type}
Best algorithm: {best_algo}
Metrics: {metrics}
Models compared: {len(comparison)}
Top features: {feature_importance[:5] if feature_importance else 'N/A'}"""

        response = llm.invoke([
            SystemMessage(content="You are a data science expert. Write a 2-3 sentence summary of the ML training results. Be concise, mention the best model, key metric, and one practical insight. Do NOT use markdown formatting."),
            HumanMessage(content=context),
        ])
        return response.content.strip()
    except Exception:
        return f"The {best_algo} model achieved the best performance among {len(comparison)} algorithms tested."


# ── Main entry point ─────────────────────────────────────────────────────────

def run_training(
    data: list[dict],
    columns: list[str],
    target_column: str | None = None,
    task_type: str = "auto",
    algorithms: str | list[str] = "auto",
    cv_folds: int = 5,
    tune_trials: int = 20,
    test_size: float = 0.2,
    model_id: str | None = None,
    conversation_id: str = "",
    dataset_id: str = "",
    progress_callback: Any = None,
) -> dict:
    """Run the full Auto ML training pipeline.

    Returns dict with all results suitable for TrainResultCard rendering.
    """
    t0 = time.time()
    log.info(">>> Training pipeline  rows=%d  cols=%d  target=%s", len(data), len(columns), target_column)

    def _emit(phase: str, **extra: Any) -> None:
        """Notify the caller of pipeline progress. Never raises."""
        if progress_callback is None:
            return
        try:
            progress_callback({"phase": phase, "elapsed": time.time() - t0, **extra})
        except Exception as exc:  # never break training because of a callback
            log.debug("progress_callback raised: %s", exc)

    _emit("preprocessing")

    # Build DataFrame
    df = pd.DataFrame(data, columns=columns) if columns else pd.DataFrame(data)

    # Step 1: Detect task type
    y_col = df[target_column] if target_column and target_column in df.columns else None
    detected_task = _detect_task_type(y_col, task_type)
    log.info("  Task type: %s", detected_task)

    # Step 2: Auto-preprocess
    prep = _auto_preprocess(df, target_column, detected_task, test_size)
    X_train = np.array(prep["X_train"])
    X_test = np.array(prep["X_test"])
    y_train = np.array(prep["y_train"]) if prep["y_train"] is not None else None
    y_test = np.array(prep["y_test"]) if prep["y_test"] is not None else None
    feature_names = prep["feature_names"]
    class_names = prep["class_names"]
    log.info("  Preprocessed: X_train=%s  X_test=%s  features=%d", X_train.shape, X_test.shape, len(feature_names))

    # Capture small references needed at Step 7 before releasing the bulky copies
    df_shape = df.shape
    log_target = bool(prep.get("log_target", False))
    pipeline_bundle = {
        "scaler": prep["scaler"],
        "encoders": prep["encoders"],
        "label_encoder": prep["label_encoder"],
        "log_target": log_target,
    }

    # Release raw DataFrame + preprocessing dict — both held duplicate copies of training data
    del df, prep
    gc.collect()

    # Step 3: Select algorithms
    if algorithms == "auto":
        n_classes = len(class_names) if class_names else None
        algo_keys = select_algorithms(detected_task, X_train.shape[0], X_train.shape[1], n_classes)
    else:
        algo_keys = algorithms if isinstance(algorithms, list) else [algorithms]

    registry = get_algorithms_for_task(detected_task)
    log.info("  Algorithms: %s", [registry[k]["display_name"] for k in algo_keys])

    _emit("cv", total_algos=len(algo_keys),
          algos=[registry[k]["display_name"] for k in algo_keys])

    # Step 4: Cross-validate
    cv_results = _cross_validate_all(algo_keys, detected_task, X_train, y_train, cv_folds)
    if not cv_results:
        _emit("error", error="All algorithms failed during cross-validation.")
        return {"success": False, "error": "All algorithms failed during cross-validation."}

    _emit("cv_done", algos_completed=len(cv_results))

    # Sort by primary metric
    def _primary_metric(r: dict) -> float:
        m = r["cv_metrics"]
        if detected_task == "classification":
            return m.get("f1_mean", m.get("accuracy_mean", 0))
        elif detected_task == "regression":
            return -m.get("rmse_mean", 1e9)
        else:
            return m.get("silhouette", 0)

    cv_results.sort(key=_primary_metric, reverse=True)

    # Step 5: Tune top 3
    top_n = min(3, len(cv_results))
    tuned_results: list[dict] = []
    for i in range(top_n):
        algo_key = cv_results[i]["algo_key"]
        display_name = cv_results[i]["display_name"]
        log.info("  Tuning %d/%d: %s  (%d trials)", i + 1, top_n, display_name, tune_trials)
        _emit("tuning", current=i + 1, total=top_n, algo=display_name)
        tuned = _tune_model(algo_key, detected_task, X_train, y_train, tune_trials, cv_folds)
        tuned_results.append({
            "algo_key": algo_key,
            "display_name": display_name,
            **tuned,
        })

    _emit("evaluation", total_algos=len(cv_results))

    # Step 6: Evaluate EVERY algorithm on the held-out test set so the comparison
    # table and the headline use the same numbers (same data, same scale).
    # This guarantees the #1 row in the table matches the "Best Model" headline.
    if detected_task in ("classification", "regression"):
        log.info("  Test-set evaluation for comparison table (%d algos)…", len(cv_results))
        for cv_entry in cv_results:
            algo_key = cv_entry["algo_key"]
            # Prefer the tuned model if this algo was in the top 3
            model = next((t["model"] for t in tuned_results if t["algo_key"] == algo_key), None)
            if model is None:
                model = instantiate_model(detected_task, algo_key)
                try:
                    if y_train is not None:
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train)
                except Exception as exc:
                    log.warning("    Fit failed for %s: %s", cv_entry["display_name"], exc)
                    continue
            try:
                y_pred_test = model.predict(X_test)
            except Exception as exc:
                log.warning("    Predict failed for %s: %s", cv_entry["display_name"], exc)
                continue

            # Keep predictions and y_test on whatever scale training used.
            # If log_target was applied, metrics stay on log scale — consistent
            # with the CV scoring metric and dataset-agnostic.
            y_test_eval = y_test

            try:
                if detected_task == "classification":
                    y_prob_test = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
                    tm = compute_classification_metrics(y_test_eval, y_pred_test, y_prob_test, class_names)
                    cv_entry["cv_metrics"] = {
                        "accuracy_mean": float(tm.get("accuracy", 0.0)),
                        "f1_mean": float(tm.get("f1", 0.0)),
                        "precision_mean": float(tm.get("precision", 0.0)),
                        "recall_mean": float(tm.get("recall", 0.0)),
                        "accuracy_std": 0.0, "f1_std": 0.0,
                        "precision_std": 0.0, "recall_std": 0.0,
                    }
                else:
                    tm = compute_regression_metrics(y_test_eval, y_pred_test, len(feature_names))
                    cv_entry["cv_metrics"] = {
                        "rmse_mean": float(tm.get("rmse", 0.0)),
                        "mae_mean": float(tm.get("mae", 0.0)),
                        "r2_mean": float(tm.get("r2", 0.0)),
                        "rmse_std": 0.0, "mae_std": 0.0, "r2_std": 0.0,
                    }
                cv_entry["model"] = model
            except Exception as exc:
                log.warning("    Metrics failed for %s: %s", cv_entry["display_name"], exc)

    # Re-sort by primary metric — now reflects true test-set performance.
    cv_results.sort(key=_primary_metric, reverse=True)

    # Best model = #1 in the comparison table (same data, same scale ⇒ matches headline)
    best_cv = cv_results[0]
    best_tuned = next((t for t in tuned_results if t["algo_key"] == best_cv["algo_key"]), None)
    if best_tuned is not None:
        best_model = best_tuned["model"]
        best_algo_key = best_tuned["algo_key"]
        best_display = best_tuned["display_name"]
        best_params = best_tuned["best_params"]
    else:
        best_model = best_cv["model"]
        best_algo_key = best_cv["algo_key"]
        best_display = best_cv["display_name"]
        best_params = {}
    log.info("  Best model: %s (primary=%.4f)", best_display, _primary_metric(best_cv))

    _emit("charts")

    # Get predictions
    if detected_task == "clustering":
        y_pred = best_model.labels_ if hasattr(best_model, "labels_") else best_model.predict(X_test)
        y_prob = None
        final_metrics = compute_clustering_metrics(X_test, y_pred)
    elif detected_task == "classification":
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
        final_metrics = compute_classification_metrics(y_test, y_pred, y_prob, class_names)
    else:
        y_pred = best_model.predict(X_test)
        y_prob = None
        # Keep on training scale (log if log_target was applied) so headline
        # metrics match the comparison table exactly — generic across datasets.
        # /predict still expm1s for human-readable predictions (predict_agent.py).
        final_metrics = compute_regression_metrics(y_test, y_pred, len(feature_names))

        # For chart interpretability (residual plot, actual-vs-predicted),
        # show values in the original unit (e.g., dollars) not log. The metrics
        # above already locked-in their log-scale numbers for the headline,
        # so mutating these is safe. Learning curve still trains on log y_train.
        if log_target:
            y_pred = np.expm1(y_pred)
            y_test = np.expm1(y_test)

    # Build comparison table
    comparison_table = []
    for i, r in enumerate(cv_results):
        row: dict[str, Any] = {
            "algorithm": r["algo_key"],
            "algorithmDisplay": r["display_name"],
            "trainTime": round(r["train_time"], 2),
            "rank": i + 1,
        }
        # Flatten cv_metrics for display
        row["metrics"] = {}
        for k, v in r["cv_metrics"].items():
            if k.endswith("_mean"):
                row["metrics"][k.replace("_mean", "")] = round(v, 4)
        comparison_table.append(row)

    # Feature importance
    feature_importance: list[dict] = []
    if hasattr(best_model, "feature_importances_"):
        imp = best_model.feature_importances_
        for fn, iv in sorted(zip(feature_names, imp), key=lambda x: -x[1]):
            feature_importance.append({"feature": fn, "importance": round(float(iv), 4)})
    elif hasattr(best_model, "coef_"):
        coef = np.abs(np.array(best_model.coef_).ravel())
        if len(coef) == len(feature_names):
            for fn, iv in sorted(zip(feature_names, coef), key=lambda x: -x[1]):
                feature_importance.append({"feature": fn, "importance": round(float(iv), 4)})

    # Charts
    charts = build_charts(
        task_type=detected_task,
        model=best_model,
        X_test=X_test,
        y_test=y_test if y_test is not None else y_pred,
        y_pred=y_pred,
        y_prob=y_prob,
        feature_names=feature_names,
        class_names=class_names or None,
        X_train=X_train,
        y_train=y_train,
    )

    # Classification report text
    clf_report = ""
    if detected_task == "classification" and class_names:
        try:
            clf_report = get_classification_report_text(y_test, y_pred, class_names)
        except ValueError:
            # Test set may have fewer classes than full set — use labels present in test
            test_classes = [class_names[i] for i in sorted(set(y_test.astype(int)))] if len(class_names) > 0 else None
            try:
                clf_report = get_classification_report_text(y_test, y_pred, test_classes or class_names)
            except Exception:
                clf_report = ""

    _emit("finalizing")

    # Step 7: Save model as draft (user must explicitly promote via /train/models/{id}/save)
    saved = save_model(
        model=best_model,
        pipeline=pipeline_bundle,
        conversation_id=conversation_id,
        dataset_id=dataset_id,
        task_type=detected_task,
        algorithm=best_algo_key,
        algorithm_display=best_display,
        target_column=target_column or "",
        feature_columns=feature_names,
        metrics={k: round(v, 4) for k, v in final_metrics.items()},
        hyperparameters=best_params,
        training_duration=time.time() - t0,
        dataset_shape=(df_shape[0], df_shape[1]),
        is_draft=True,
    )

    # AI summary
    ai_summary = _generate_ai_summary(
        detected_task, best_display, final_metrics, comparison_table, feature_importance, model_id,
    )

    duration = time.time() - t0
    log.info("<<< Training done in %.1fs  best=%s", duration, best_display)

    return {
        "success": True,
        "model_id": saved["model_id"],
        "best_algorithm": best_algo_key,
        "best_algorithm_display": best_display,
        "task_type": detected_task,
        "target_column": target_column or "",
        "metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "comparison_table": comparison_table,
        "charts": charts,
        "feature_importance": feature_importance,
        "classification_report": clf_report,
        "ai_summary": ai_summary,
        "download_url": f"/train/models/{saved['model_id']}/download",
        "dataset_shape": [df_shape[0], df_shape[1]],
        "training_duration": round(duration, 1),
    }

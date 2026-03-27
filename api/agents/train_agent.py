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
    """Minimal preprocessing for training: nulls, encoding, scaling, split."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_column], errors="ignore") if target_column else df.copy()
    y = df[target_column].copy() if target_column and target_column in df.columns else None

    # Coerce string-encoded numbers back to numeric (JSON deserialization artifact)
    if y is not None and y.dtype == "object":
        coerced = pd.to_numeric(y, errors="coerce")
        if coerced.notna().sum() / max(len(y), 1) > 0.8:
            y = coerced
            y = y.fillna(y.median())

    # Coerce string-encoded numeric feature columns
    for col in X.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(X[col], errors="coerce")
        if coerced.notna().sum() / max(len(X), 1) > 0.8:
            X[col] = coerced

    # Drop all-null columns
    null_cols = X.columns[X.isnull().all()].tolist()
    if null_cols:
        X = X.drop(columns=null_cols)

    # Fill nulls
    for col in X.select_dtypes(include="number").columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")

    # Encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders: dict[str, OrdinalEncoder] = {}
    for col in cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[col] = enc.fit_transform(X[[col]])
        encoders[col] = enc

    # Encode target for classification
    label_encoder = None
    class_names: list[str] = []
    if y is not None and task_type == "classification":
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), name=y.name)
        class_names = [str(c) for c in label_encoder.classes_]
    elif y is not None:
        y = y.astype(float)

    feature_names = X.columns.tolist()

    # Scale
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_names, index=X.index)

    # Train/test split (fall back to non-stratified if classes have < 2 members)
    if y is not None:
        stratify = y if task_type == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=stratify,
            )
        except ValueError:
            log.warning("Stratified split failed (singleton classes) — falling back to random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
            )
    else:
        X_train, X_test = X_scaled, X_scaled
        y_train, y_test = None, None

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "encoders": encoders,
    }


# ── Step 4: Cross-validate ───────────────────────────────────────────────────

def _cross_validate_all(
    algo_keys: list[str],
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_folds: int = 5,
) -> list[dict]:
    """Run cross-validation for all selected algorithms."""
    results: list[dict] = []
    registry = get_algorithms_for_task(task_type)

    for algo_key in algo_keys:
        entry = registry[algo_key]
        model = instantiate_model(task_type, algo_key)
        t0 = time.time()

        try:
            if task_type == "classification":
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = {"accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted", "recall": "recall_weighted"}
            elif task_type == "regression":
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
            else:
                # Clustering: no CV, just fit + score
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

            cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            duration = time.time() - t0

            cv_metrics: dict[str, float] = {}
            for metric_name, scorer_name in scoring.items():
                vals = cv_results[f"test_{metric_name}"]
                if metric_name in ("rmse", "mae"):
                    vals = -vals  # neg_ scorers
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

def _tune_model(
    algo_key: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 5,
) -> dict:
    """Tune hyperparameters with Optuna for a single algorithm."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    registry = get_algorithms_for_task(task_type)
    entry = registry[algo_key]
    tunable = entry.get("tunable_params", {})

    if not tunable:
        model = instantiate_model(task_type, algo_key)
        if y_train is not None:
            model.fit(X_train, y_train)
        else:
            model.fit(X_train)
        return {"model": model, "best_params": {}, "best_score": 0.0}

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

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "f1_weighted"
        elif task_type == "regression":
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scoring = "neg_root_mean_squared_error"
        else:
            model.fit(X_train)
            labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_train)
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if n_labels < 2:
                return -1.0
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(X_train, labels))

        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_model = instantiate_model(task_type, algo_key, best_params)
    if y_train is not None:
        best_model.fit(X_train, y_train)
    else:
        best_model.fit(X_train)

    return {"model": best_model, "best_params": best_params, "best_score": study.best_value}


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
    tune_trials: int = 50,
    test_size: float = 0.2,
    model_id: str | None = None,
    conversation_id: str = "",
    dataset_id: str = "",
) -> dict:
    """Run the full Auto ML training pipeline.

    Returns dict with all results suitable for TrainResultCard rendering.
    """
    t0 = time.time()
    log.info(">>> Training pipeline  rows=%d  cols=%d  target=%s", len(data), len(columns), target_column)

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

    # Step 3: Select algorithms
    if algorithms == "auto":
        n_classes = len(class_names) if class_names else None
        algo_keys = select_algorithms(detected_task, X_train.shape[0], X_train.shape[1], n_classes)
    else:
        algo_keys = algorithms if isinstance(algorithms, list) else [algorithms]

    registry = get_algorithms_for_task(detected_task)
    log.info("  Algorithms: %s", [registry[k]["display_name"] for k in algo_keys])

    # Step 4: Cross-validate
    cv_results = _cross_validate_all(algo_keys, detected_task, X_train, y_train, cv_folds)
    if not cv_results:
        return {"success": False, "error": "All algorithms failed during cross-validation."}

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
        tuned = _tune_model(algo_key, detected_task, X_train, y_train, tune_trials, cv_folds)
        tuned_results.append({
            "algo_key": algo_key,
            "display_name": display_name,
            **tuned,
        })

    # Step 6: Final evaluation — pick best tuned model
    best_tuned = max(tuned_results, key=lambda r: r["best_score"])
    best_model = best_tuned["model"]
    best_algo_key = best_tuned["algo_key"]
    best_display = best_tuned["display_name"]
    best_params = best_tuned["best_params"]
    log.info("  Best model: %s (score=%.4f)", best_display, best_tuned["best_score"])

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
        final_metrics = compute_regression_metrics(y_test, y_pred, len(feature_names))

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
        clf_report = get_classification_report_text(y_test, y_pred, class_names)

    # Step 7: Save model
    pipeline_bundle = {
        "scaler": prep["scaler"],
        "encoders": prep["encoders"],
        "label_encoder": prep["label_encoder"],
    }
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
        dataset_shape=(df.shape[0], df.shape[1]),
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
        "dataset_shape": [df.shape[0], df.shape[1]],
        "training_duration": round(duration, 1),
    }

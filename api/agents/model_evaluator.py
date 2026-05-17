"""Model Evaluator — compute metrics and generate Plotly evaluation charts.

Supports classification (binary + multiclass), regression, and clustering.
All charts are returned as Plotly JSON dicts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)

from api.logger import get_logger

log = get_logger(__name__)

# PrepPilot Plotly theme
_THEME = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#0A0A0A"),
    margin=dict(l=50, r=30, t=50, b=50),
    colorway=["#E8500A", "#FF6B2B", "#1E88E5", "#43A047", "#8E24AA",
              "#F4511E", "#00ACC1", "#FFB300", "#5E35B1", "#D81B60"],
)


# ── Classification Metrics ───────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute all classification metrics."""
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None:
        try:
            if n_classes == 2:
                prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                metrics["auc_roc"] = float(roc_auc_score(y_true, prob_col))
            else:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except (ValueError, IndexError):
            pass

    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_features: int = 1) -> dict[str, float]:
    """Compute all regression metrics."""
    n = len(y_true)
    r2 = float(r2_score(y_true, y_pred))
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    mape_val = float(mean_absolute_percentage_error(y_true, y_pred))

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": r2,
        "adjusted_r2": adj_r2,
        "mape": mape_val,
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def compute_clustering_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute clustering quality metrics."""
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

    n_labels = len(set(labels)) - (1 if -1 in labels else 0)
    metrics: dict[str, float] = {}

    if n_labels >= 2:
        valid = labels != -1
        if valid.sum() > n_labels:
            metrics["silhouette"] = float(silhouette_score(X[valid], labels[valid]))
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(X[valid], labels[valid]))
            metrics["davies_bouldin"] = float(davies_bouldin_score(X[valid], labels[valid]))

    metrics["n_clusters"] = float(n_labels)
    if -1 in labels:
        metrics["noise_points"] = float((labels == -1).sum())

    return metrics


# ── Classification Charts ────────────────────────────────────────────────────

def chart_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = [str(c) for c in class_names]

    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#FFF3E0"], [1, "#E8500A"]],
        text=cm, texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted", yaxis_title="Actual",
        yaxis_autorange="reversed",
        **_THEME,
    )
    return fig.to_dict()


def chart_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    """ROC curve (binary or multiclass OVR)."""
    fig = go.Figure()
    n_classes = len(class_names)

    if n_classes == 2:
        prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        fpr, tpr, _ = roc_curve(y_true, prob_col)
        auc_val = roc_auc_score(y_true, prob_col)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_val:.3f}", line=dict(color="#E8500A", width=2)))
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = ["#E8500A", "#FF6B2B", "#1E88E5", "#43A047", "#8E24AA",
                  "#F4511E", "#00ACC1", "#FFB300", "#5E35B1", "#D81B60"]
        for i in range(min(n_classes, 10)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            auc_val = roc_auc_score(y_bin[:, i], y_prob[:, i])
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{class_names[i]} (AUC={auc_val:.3f})",
                line=dict(color=colors[i % len(colors)], width=2),
            ))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#CCCCCC"), showlegend=False))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", **_THEME)
    return fig.to_dict()


def chart_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict:
    """Precision-Recall curve."""
    fig = go.Figure()
    n_classes = len(class_names)

    if n_classes == 2:
        prob_col = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        prec, rec, _ = precision_recall_curve(y_true, prob_col)
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR Curve", line=dict(color="#E8500A", width=2)))
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = ["#E8500A", "#FF6B2B", "#1E88E5", "#43A047", "#8E24AA",
                  "#F4511E", "#00ACC1", "#FFB300", "#5E35B1", "#D81B60"]
        for i in range(min(n_classes, 10)):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
            fig.add_trace(go.Scatter(
                x=rec, y=prec, mode="lines",
                name=class_names[i],
                line=dict(color=colors[i % len(colors)], width=2),
            ))

    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision", **_THEME)
    return fig.to_dict()


def chart_feature_importance(feature_names: list[str], importances: np.ndarray, top_n: int = 20) -> dict:
    """Feature importance bar chart (top N)."""
    idx = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color="#E8500A",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(title=f"Feature Importance (Top {min(top_n, len(feature_names))})", xaxis_title="Importance", **_THEME)
    return fig.to_dict()


def chart_learning_curve(model: Any, X: np.ndarray, y: np.ndarray, task_type: str, cv: int = 5) -> dict:
    """Learning curve — train vs validation performance over dataset size."""
    from sklearn.model_selection import learning_curve

    scoring = "f1_weighted" if task_type == "classification" else "neg_root_mean_squared_error"
    sizes = np.linspace(0.2, 1.0, 5)

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=sizes, cv=cv, scoring=scoring, n_jobs=2,
        )
    except Exception:
        # Fallback: skip learning curve on error
        return {}

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    if task_type == "regression":
        train_mean, val_mean = -train_mean, -val_mean
        train_std, val_std = train_std, val_std

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean, mode="lines+markers", name="Training",
        line=dict(color="#E8500A", width=2),
        error_y=dict(type="data", array=train_std, visible=True),
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean, mode="lines+markers", name="Validation",
        line=dict(color="#1E88E5", width=2),
        error_y=dict(type="data", array=val_std, visible=True),
    ))

    y_label = "F1 Score" if task_type == "classification" else "RMSE"
    fig.update_layout(title="Learning Curve", xaxis_title="Training Size", yaxis_title=y_label, **_THEME)
    return fig.to_dict()


# ── Regression Charts ────────────────────────────────────────────────────────

def chart_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Scatter: actual vs predicted with perfect-prediction line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color="#E8500A", size=5, opacity=0.6),
        name="Predictions",
        hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", line=dict(dash="dash", color="#CCCCCC"), showlegend=False))
    fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted", **_THEME)
    return fig.to_dict()


def chart_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Residual scatter plot."""
    residuals = y_true - y_pred
    fig = go.Figure(go.Scatter(
        x=y_pred, y=residuals, mode="markers",
        marker=dict(color="#E8500A", size=5, opacity=0.6),
        hovertemplate="Predicted: %{x:.2f}<br>Residual: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#CCCCCC")
    fig.update_layout(title="Residual Plot", xaxis_title="Predicted", yaxis_title="Residual", **_THEME)
    return fig.to_dict()


def chart_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Residual distribution histogram."""
    residuals = y_true - y_pred
    fig = go.Figure(go.Histogram(
        x=residuals, nbinsx=30,
        marker_color="#E8500A", opacity=0.8,
        hovertemplate="Residual: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(title="Residual Distribution", xaxis_title="Residual", yaxis_title="Count", **_THEME)
    return fig.to_dict()


# ── Clustering Charts ────────────────────────────────────────────────────────

def chart_cluster_scatter(X: np.ndarray, labels: np.ndarray) -> dict:
    """PCA 2D scatter of clusters."""
    from sklearn.decomposition import PCA

    X_2d = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X[:, :2]
    unique_labels = sorted(set(labels))
    colors = ["#E8500A", "#1E88E5", "#43A047", "#8E24AA", "#FFB300",
              "#F4511E", "#00ACC1", "#5E35B1", "#D81B60", "#FF6B2B"]

    fig = go.Figure()
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = "Noise" if lbl == -1 else f"Cluster {lbl}"
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], mode="markers",
            marker=dict(color="#CCCCCC" if lbl == -1 else colors[i % len(colors)], size=5, opacity=0.7),
            name=name,
        ))
    fig.update_layout(title="Cluster Visualization (PCA 2D)", xaxis_title="PC1", yaxis_title="PC2", **_THEME)
    return fig.to_dict()


def chart_cluster_sizes(labels: np.ndarray) -> dict:
    """Bar chart of cluster sizes."""
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    names = [f"Cluster {u}" for u in unique]

    n = len(names)
    colors = ["#E8500A", "#1E88E5", "#43A047", "#8E24AA", "#FFB300",
              "#F4511E", "#00ACC1", "#5E35B1", "#D81B60", "#FF6B2B"]
    bar_colors = [colors[i % len(colors)] for i in range(n)]

    fig = go.Figure(go.Bar(x=names, y=counts, marker_color=bar_colors))
    fig.update_layout(title="Cluster Size Distribution", xaxis_title="Cluster", yaxis_title="Count", **_THEME)
    return fig.to_dict()


# ── Chart builder (collects all relevant charts) ─────────────────────────────

def build_charts(
    task_type: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    feature_names: list[str],
    class_names: list[str] | None = None,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> list[dict]:
    """Build all evaluation charts for a given task type.

    Returns list of {"name": str, "plotly_json": dict}.
    """
    charts: list[dict] = []

    if task_type == "classification" and class_names:
        charts.append({"name": "Confusion Matrix", "plotly_json": chart_confusion_matrix(y_test, y_pred, class_names)})

        if y_prob is not None:
            try:
                charts.append({"name": "ROC Curve", "plotly_json": chart_roc_curve(y_test, y_prob, class_names)})
            except Exception:
                pass
            try:
                charts.append({"name": "Precision-Recall Curve", "plotly_json": chart_precision_recall(y_test, y_prob, class_names)})
            except Exception:
                pass

    elif task_type == "regression":
        charts.append({"name": "Actual vs Predicted", "plotly_json": chart_actual_vs_predicted(y_test, y_pred)})
        charts.append({"name": "Residual Plot", "plotly_json": chart_residuals(y_test, y_pred)})
        charts.append({"name": "Residual Distribution", "plotly_json": chart_residual_distribution(y_test, y_pred)})

    elif task_type == "clustering":
        charts.append({"name": "Cluster Visualization", "plotly_json": chart_cluster_scatter(X_test, y_pred)})
        charts.append({"name": "Cluster Size Distribution", "plotly_json": chart_cluster_sizes(y_pred)})

    # Feature importance (works for tree-based models)
    if hasattr(model, "feature_importances_"):
        imp = np.array(model.feature_importances_)
        charts.append({"name": "Feature Importance", "plotly_json": chart_feature_importance(feature_names, imp)})
    elif hasattr(model, "coef_"):
        coef = np.abs(np.array(model.coef_).ravel())
        if len(coef) == len(feature_names):
            charts.append({"name": "Feature Importance", "plotly_json": chart_feature_importance(feature_names, coef)})

    # Learning curve (only if we have training data)
    if X_train is not None and y_train is not None and task_type in ("classification", "regression"):
        lc = chart_learning_curve(model, X_train, y_train, task_type)
        if lc:
            charts.append({"name": "Learning Curve", "plotly_json": lc})

    return charts


def get_classification_report_text(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> str:
    """Return sklearn classification_report as formatted text."""
    return classification_report(y_true, y_pred, target_names=[str(c) for c in class_names], zero_division=0)

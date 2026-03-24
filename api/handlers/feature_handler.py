"""Feature engineering handler — importance, PCA, correlation filter, log transform."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


class FeatureHandler(BaseHandler):

    @staticmethod
    def handle_feature_importance(df: pd.DataFrame, params: dict) -> HandlerResult:
        target = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not target or target not in df.columns:
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No target column specified or found")
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            X = df[num_cols].drop(columns=[target], errors="ignore").dropna()
            y = df.loc[X.index, target]
            if y.nunique() <= 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
            imp = imp.sort_values("importance", ascending=True)
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Feature Importance")
            return HandlerResult(success=True, result_df=imp, charts_plotly=[fig.to_json()],
                                 summary=f"Feature importance (target='{target}', model=RandomForest)")
        except Exception as e:
            return HandlerResult(success=False, error=f"Feature importance error: {e}")

    @staticmethod
    def handle_pca(df: pd.DataFrame, params: dict) -> HandlerResult:
        n_components = params.get("n", 2)
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            num_cols = df.select_dtypes(include="number").columns.tolist()
            X = df[num_cols].dropna()
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=min(n_components, len(num_cols)))
            components = pca.fit_transform(X_scaled)
            col_names = [f"PC{i+1}" for i in range(components.shape[1])]
            result = pd.DataFrame(components, columns=col_names, index=X.index)
            var_explained = pca.explained_variance_ratio_
            fig = px.scatter(result, x="PC1", y="PC2",
                             title=f"PCA (PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})")
            return HandlerResult(success=True, result_df=result, charts_plotly=[fig.to_json()],
                                 output_type="generate",
                                 summary=f"PCA: {len(num_cols)} features → {n_components} components, {sum(var_explained):.1%} variance")
        except Exception as e:
            return HandlerResult(success=False, error=f"PCA error: {e}")

    @staticmethod
    def handle_correlation_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
        threshold = params.get("value", 0.95)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return HandlerResult(success=False, error="Need at least 2 numeric columns")
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = [c for c in upper.columns if upper[c].max() > threshold]
        result = df.drop(columns=to_drop)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Dropped {len(to_drop)} columns with correlation > {threshold}: {to_drop}")

    @staticmethod
    def handle_log_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            result[f"{col}_log"] = np.log1p(result[col])
            summary = f"Log-transformed '{col}' → '{col}_log'"
        else:
            skewed = []
            for c in result.select_dtypes(include="number").columns:
                if abs(result[c].skew()) > 1 and (result[c] >= 0).all():
                    result[f"{c}_log"] = np.log1p(result[c])
                    skewed.append(c)
            summary = f"Log-transformed {len(skewed)} skewed columns: {skewed}"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    @staticmethod
    def handle_variance_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
        threshold = params.get("value", 0.01)
        num_cols = df.select_dtypes(include="number").columns
        variances = df[num_cols].var()
        low_var = variances[variances < threshold].index.tolist()
        result = df.drop(columns=low_var)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Dropped {len(low_var)} low-variance columns (threshold={threshold}): {low_var}")

    @staticmethod
    def handle_polynomial_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c in num_cols] or num_cols[:3]
        result = df.copy()
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                result[f"{c1}_x_{c2}"] = result[c1] * result[c2]
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Added interaction features for {len(cols)} columns")

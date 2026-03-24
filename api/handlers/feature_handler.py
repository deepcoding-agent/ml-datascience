"""Feature engineering handler — importance, PCA, correlation filter, log transform."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.viz_handler import _style
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
            fig = px.bar(imp, x="importance", y="feature", orientation="h")
            fig.update_traces(marker_color="#FB8C3C")
            _style(fig, title="Feature Importance")
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
            fig = px.scatter(result, x="PC1", y="PC2", opacity=0.7)
            _style(fig, title=f"PCA (PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%})")
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

    # ── 1. Datetime feature extraction (most important) ──────────────────────

    @staticmethod
    def handle_datetime_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract year, month, day, dayofweek, hour from datetime columns."""
        col = params.get("column")
        result = df.copy()

        # Find datetime columns
        dt_cols = result.select_dtypes(include="datetime").columns.tolist()
        if col and col in result.columns and col not in dt_cols:
            try:
                result[col] = pd.to_datetime(result[col], errors="coerce", format="mixed")
                dt_cols = [col]
            except Exception:
                return HandlerResult(success=False, error=f"Cannot parse '{col}' as datetime")
        elif col and col in dt_cols:
            dt_cols = [col]

        if not dt_cols:
            return HandlerResult(success=False, error="No datetime columns found")

        created = []
        for c in dt_cols:
            dt = result[c].dt
            result[f"{c}_year"] = dt.year
            result[f"{c}_month"] = dt.month
            result[f"{c}_day"] = dt.day
            result[f"{c}_dayofweek"] = dt.dayofweek
            if dt.hour.max() > 0:
                result[f"{c}_hour"] = dt.hour
                created.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek", f"{c}_hour"])
            else:
                created.extend([f"{c}_year", f"{c}_month", f"{c}_day", f"{c}_dayofweek"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted {len(created)} datetime features from {len(dt_cols)} column(s)",
        )

    # ── 2. Target encoding ───────────────────────────────────────────────────

    @staticmethod
    def handle_target_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mean/target encoding: replace categorical values with the mean of the target."""
        target = params.get("target") or params.get("column")
        col = params.get("encode_column")
        if not target or target not in df.columns:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No target column found")

        result = df.copy()
        cat_cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        encoded = []

        global_mean = result[target].mean()
        for c in cat_cols:
            means = result.groupby(c)[target].mean()
            result[f"{c}_target_enc"] = result[c].map(means).fillna(global_mean)
            encoded.append(c)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Target-encoded {len(encoded)} columns using target='{target}'",
        )

    # ── 3. Select K best features ────────────────────────────────────────────

    @staticmethod
    def handle_select_k_best(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Select top K features using statistical tests (f_classif or f_regression)."""
        target = params.get("column")
        k = params.get("k", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not target or target not in df.columns:
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No target column found")

        try:
            from sklearn.feature_selection import SelectKBest, f_classif, f_regression
            feature_cols = [c for c in num_cols if c != target]
            X = df[feature_cols].dropna()
            y = df.loc[X.index, target]

            scorer = f_classif if y.nunique() <= 10 else f_regression
            k_actual = min(k, len(feature_cols))
            selector = SelectKBest(scorer, k=k_actual)
            selector.fit(X, y)

            scores = pd.DataFrame({
                "feature": feature_cols,
                "score": selector.scores_,
                "selected": selector.get_support(),
            }).sort_values("score", ascending=True)

            fig = px.bar(scores, x="score", y="feature", orientation="h",
                         color="selected", color_discrete_map={True: "#FB8C3C", False: "#E0E0E0"})
            _style(fig, title=f"Top {k_actual} Features", showlegend=False)

            selected = scores[scores["selected"]]["feature"].tolist()
            result = df[selected + [target]]
            return HandlerResult(
                success=True, result_df=result, charts_plotly=[fig.to_json()],
                output_type="generate",
                summary=f"Selected top {k_actual} features (target='{target}'): {selected}",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"SelectKBest error: {e}")

    # ── 4. Power transform (Box-Cox / Yeo-Johnson) ──────────────────────────

    @staticmethod
    def handle_power_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply Yeo-Johnson or Box-Cox power transform to normalize distributions."""
        col = params.get("column")
        method = params.get("method", "yeo-johnson")
        result = df.copy()

        try:
            from sklearn.preprocessing import PowerTransformer
            cols = [col] if col and col in result.columns else result.select_dtypes(include="number").columns.tolist()
            X = result[cols].dropna()

            pt = PowerTransformer(method=method, standardize=True)
            transformed = pd.DataFrame(pt.fit_transform(X), index=X.index, columns=cols)
            for c in cols:
                result[c] = result[c].astype(float)
            result.loc[X.index, cols] = transformed

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Power transform ({method}) on {len(cols)} columns",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Power transform error: {e}")

    # ── 5. Ratio features ────────────────────────────────────────────────────

    @staticmethod
    def handle_ratio_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create ratio features (col_a / col_b) for numeric column pairs."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c in num_cols] or num_cols[:4]
        result = df.copy()
        created = []

        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                safe = result[c2].replace(0, np.nan)
                result[f"{c1}_per_{c2}"] = (result[c1] / safe).round(4)
                created.append(f"{c1}_per_{c2}")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} ratio features",
        )

    # ── 6. Frequency encoding ────────────────────────────────────────────────

    @staticmethod
    def handle_frequency_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode categorical values by their frequency (count or normalized)."""
        col = params.get("column")
        result = df.copy()
        cat_cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        encoded = []

        for c in cat_cols:
            freq = result[c].value_counts(normalize=True)
            result[f"{c}_freq"] = result[c].map(freq).round(4)
            encoded.append(c)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Frequency-encoded {len(encoded)} columns",
        )

    # ── 7. Cyclical encoding (sin/cos) ───────────────────────────────────────

    @staticmethod
    def handle_cyclical_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode cyclical features (month, dayofweek, hour) using sin/cos."""
        col = params.get("column")
        period = params.get("period")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")

        result = df.copy()
        values = pd.to_numeric(result[col], errors="coerce")

        if period is None:
            max_val = values.max()
            period = int(max_val) + 1 if max_val and max_val > 0 else 12

        result[f"{col}_sin"] = np.sin(2 * np.pi * values / period).round(4)
        result[f"{col}_cos"] = np.cos(2 * np.pi * values / period).round(4)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cyclical encoding '{col}' (period={period}) → {col}_sin, {col}_cos",
        )

    # ── 8. Square root transform ─────────────────────────────────────────────

    @staticmethod
    def handle_sqrt_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply square root transform to numeric columns (good for moderate skew)."""
        col = params.get("column")
        result = df.copy()

        if col and col in result.columns:
            result[f"{col}_sqrt"] = np.sqrt(result[col].clip(lower=0))
            summary = f"Sqrt-transformed '{col}' → '{col}_sqrt'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                if abs(result[c].skew()) > 0.5 and (result[c] >= 0).all():
                    result[f"{c}_sqrt"] = np.sqrt(result[c])
                    transformed.append(c)
            summary = f"Sqrt-transformed {len(transformed)} columns: {transformed}"

        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 9. Mutual information ────────────────────────────────────────────────

    @staticmethod
    def handle_mutual_info(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mutual information scores — works for both numeric and categorical features."""
        target = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()

        if not target or target not in df.columns:
            target = num_cols[-1] if num_cols else None
        if target is None:
            return HandlerResult(success=False, error="No target column found")

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            feature_cols = [c for c in num_cols if c != target]
            clean_df = df[feature_cols + [target]].dropna()
            X = clean_df[feature_cols]
            y = clean_df[target]

            scorer = mutual_info_classif if y.nunique() <= 10 else mutual_info_regression
            mi_scores = scorer(X, y, random_state=42)

            result = pd.DataFrame({
                "feature": feature_cols,
                "mutual_info": np.round(mi_scores, 4),
            }).sort_values("mutual_info", ascending=True).reset_index(drop=True)

            fig = px.bar(result, x="mutual_info", y="feature", orientation="h")
            fig.update_traces(marker_color="#FB8C3C")
            _style(fig, title=f"Mutual Information (target: {target})")

            return HandlerResult(
                success=True, result_df=result, charts_plotly=[fig.to_json()],
                summary=f"Mutual information for {len(feature_cols)} features (target='{target}')",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Mutual info error: {e}")

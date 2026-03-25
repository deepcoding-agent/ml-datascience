"""Feature engineering handler — 50 handlers for selection, transforms, encoding, creation."""
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

    # ── 10. Lag features (time series) ─────────────────────────────────────

    @staticmethod
    def handle_lag_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create lag/shift features for time series data."""
        col = params.get("column")
        lags = params.get("lags", [1, 2, 3])
        if isinstance(lags, int):
            lags = list(range(1, lags + 1))

        result = df.copy()
        cols_to_lag = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()[:3]
        )
        created = []

        for c in cols_to_lag:
            for lag in lags:
                result[f"{c}_lag{lag}"] = result[c].shift(lag)
                created.append(f"{c}_lag{lag}")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} lag features from {len(cols_to_lag)} column(s), lags={lags}",
        )

    # ── 11. Text features ──────────────────────────────────────────────────

    @staticmethod
    def handle_text_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract text statistics: length, word_count, digit_count, uppercase_ratio."""
        col = params.get("column")
        result = df.copy()
        str_cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="object").columns.tolist()
        )
        if not str_cols:
            return HandlerResult(success=False, error="No string columns found")

        created = []
        for c in str_cols:
            s = result[c].fillna("").astype(str)
            result[f"{c}_len"] = s.str.len()
            result[f"{c}_word_count"] = s.str.split().str.len().fillna(0).astype(int)
            result[f"{c}_digit_count"] = s.str.count(r"\d")
            str_len = s.str.len().replace(0, 1)
            result[f"{c}_upper_ratio"] = (s.str.count(r"[A-Z]") / str_len).round(4)
            created.extend([f"{c}_len", f"{c}_word_count", f"{c}_digit_count", f"{c}_upper_ratio"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted {len(created)} text features from {len(str_cols)} column(s)",
        )

    # ── 12. Quantile transform ─────────────────────────────────────────────

    @staticmethod
    def handle_quantile_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Quantile transform — maps values to uniform or normal distribution."""
        col = params.get("column")
        distribution = params.get("distribution", "normal")
        result = df.copy()

        try:
            from sklearn.preprocessing import QuantileTransformer
            cols = (
                [col] if col and col in result.columns
                else result.select_dtypes(include="number").columns.tolist()
            )
            X = result[cols].dropna()
            if len(X) < 2:
                return HandlerResult(success=False, error="Not enough data for quantile transform")

            qt = QuantileTransformer(output_distribution=distribution, random_state=42)
            transformed = pd.DataFrame(qt.fit_transform(X), index=X.index, columns=cols)
            for c in cols:
                result[c] = result[c].astype(float)
            result.loc[X.index, cols] = transformed

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Quantile transform ({distribution}) on {len(cols)} columns",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Quantile transform error: {e}")

    # ── 13. Diff features (time series) ────────────────────────────────────

    @staticmethod
    def handle_diff_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create difference features (first/second order) for time series."""
        col = params.get("column")
        periods = params.get("periods", 1)
        result = df.copy()

        cols_to_diff = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()[:3]
        )
        created = []

        for c in cols_to_diff:
            result[f"{c}_diff{periods}"] = result[c].diff(periods)
            created.append(f"{c}_diff{periods}")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} diff features (periods={periods})",
        )

    # ── 14. Aggregation features ───────────────────────────────────────────

    @staticmethod
    def handle_aggregation_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create group-by aggregation features (mean/std/count per group)."""
        group_col = params.get("column")
        agg_col = params.get("agg_column")
        result = df.copy()

        cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = result.select_dtypes(include="number").columns.tolist()

        if not group_col or group_col not in result.columns:
            group_col = cat_cols[0] if cat_cols else None
        if group_col is None:
            return HandlerResult(success=False, error="No categorical column found for grouping")

        agg_cols = [agg_col] if agg_col and agg_col in num_cols else num_cols[:3]
        created = []

        for c in agg_cols:
            for agg in ["mean", "std", "count"]:
                col_name = f"{c}_{agg}_by_{group_col}"
                result[col_name] = result.groupby(group_col)[c].transform(agg)
                if agg != "count":
                    result[col_name] = result[col_name].round(4)
                created.append(col_name)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} aggregation features grouped by '{group_col}'",
        )

    # ── 15. Log1p transform ──────────────────────────────────────────────────

    @staticmethod
    def handle_log1p_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply log(1+x) transform — safe for zero values, good for right-skewed data."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            result[f"{col}_log1p"] = np.log1p(result[col].clip(lower=0))
            summary = f"log1p-transformed '{col}' → '{col}_log1p'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                if (result[c] >= 0).all():
                    result[f"{c}_log1p"] = np.log1p(result[c])
                    transformed.append(c)
            summary = f"log1p-transformed {len(transformed)} non-negative columns: {transformed}"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 16. Interaction features ─────────────────────────────────────────────

    @staticmethod
    def handle_interaction_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Multiply pairs of numeric columns to create interaction features."""
        cols = params.get("columns", [])
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c in num_cols] or num_cols[:4]
        result = df.copy()
        created = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                name = f"{c1}_x_{c2}"
                result[name] = result[c1] * result[c2]
                created.append(name)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} interaction features from {len(cols)} columns",
        )

    # ── 17. Bin numeric ──────────────────────────────────────────────────────

    @staticmethod
    def handle_bin_numeric(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bin numeric column into custom bins with optional labels."""
        col = params.get("column")
        n_bins = params.get("n", 5)
        labels = params.get("labels")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        try:
            if labels and len(labels) == n_bins:
                result[f"{col}_bin"] = pd.cut(result[col], bins=n_bins, labels=labels)
            else:
                result[f"{col}_bin"] = pd.cut(result[col], bins=n_bins)
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Binned '{col}' into {n_bins} bins → '{col}_bin'",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Bin numeric error: {e}")

    # ── 18. Absolute value transform ─────────────────────────────────────────

    @staticmethod
    def handle_abs_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply absolute value transform to numeric columns."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            result[f"{col}_abs"] = result[col].abs()
            summary = f"Absolute value of '{col}' → '{col}_abs'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                if (result[c] < 0).any():
                    result[f"{c}_abs"] = result[c].abs()
                    transformed.append(c)
            summary = f"Absolute-value transformed {len(transformed)} columns with negatives"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 19. Reciprocal transform ─────────────────────────────────────────────

    @staticmethod
    def handle_reciprocal_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply 1/x reciprocal transform (zeros become NaN)."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            safe = result[col].replace(0, np.nan)
            result[f"{col}_reciprocal"] = (1.0 / safe).round(6)
            summary = f"Reciprocal of '{col}' → '{col}_reciprocal'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                safe = result[c].replace(0, np.nan)
                result[f"{c}_reciprocal"] = (1.0 / safe).round(6)
                transformed.append(c)
            summary = f"Reciprocal-transformed {len(transformed)} numeric columns"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 20. Count encode ─────────────────────────────────────────────────────

    @staticmethod
    def handle_count_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode categorical values by their absolute count."""
        col = params.get("column")
        result = df.copy()
        cat_cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        encoded = []
        for c in cat_cols:
            counts = result[c].value_counts()
            result[f"{c}_count"] = result[c].map(counts)
            encoded.append(c)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Count-encoded {len(encoded)} columns",
        )

    # ── 21. Rare category encode ─────────────────────────────────────────────

    @staticmethod
    def handle_rare_category_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Group rare categories (< N occurrences) into 'Other'."""
        col = params.get("column")
        min_count = params.get("min_count", 5)
        result = df.copy()
        cat_cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        modified = []
        for c in cat_cols:
            counts = result[c].value_counts()
            rare = counts[counts < min_count].index
            if len(rare) > 0:
                result[c] = result[c].where(~result[c].isin(rare), "Other")
                modified.append(f"{c} ({len(rare)} rare)")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Grouped rare categories (< {min_count}) into 'Other': {modified}",
        )

    # ── 22. Is-null features ─────────────────────────────────────────────────

    @staticmethod
    def handle_is_null_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create binary is_null indicator columns for each column with nulls."""
        result = df.copy()
        created = []
        cols_with_nulls = [c for c in result.columns if result[c].isnull().any()]
        for c in cols_with_nulls:
            result[f"{c}_is_null"] = result[c].isnull().astype(int)
            created.append(f"{c}_is_null")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} is_null indicator columns",
        )

    # ── 23. Is-zero features ─────────────────────────────────────────────────

    @staticmethod
    def handle_is_zero_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create binary is_zero indicator columns for numeric columns."""
        result = df.copy()
        created = []
        for c in result.select_dtypes(include="number").columns:
            if (result[c] == 0).any():
                result[f"{c}_is_zero"] = (result[c] == 0).astype(int)
                created.append(f"{c}_is_zero")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} is_zero indicator columns",
        )

    # ── 24. Rank transform ───────────────────────────────────────────────────

    @staticmethod
    def handle_rank_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Rank values as percent rank (0-1) for numeric columns."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            result[f"{col}_pctrank"] = result[col].rank(pct=True).round(4)
            summary = f"Percent-ranked '{col}' → '{col}_pctrank'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                result[f"{c}_pctrank"] = result[c].rank(pct=True).round(4)
                transformed.append(c)
            summary = f"Percent-ranked {len(transformed)} numeric columns"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 25. KBins discretize ─────────────────────────────────────────────────

    @staticmethod
    def handle_kbins_discretize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """KBins discretizer (uniform/quantile/kmeans) via sklearn."""
        col = params.get("column")
        n_bins = params.get("n", 5)
        strategy = params.get("strategy", "quantile")
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            result = df.copy()
            cols = (
                [col] if col and col in result.columns
                else result.select_dtypes(include="number").columns.tolist()
            )
            X = result[cols].dropna()
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
            binned = pd.DataFrame(kbd.fit_transform(X), index=X.index, columns=[f"{c}_kbin" for c in cols])
            for c in binned.columns:
                result[c] = np.nan
                result.loc[X.index, c] = binned[c].astype(int)
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"KBins discretized {len(cols)} columns (n={n_bins}, strategy={strategy})",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"KBins discretize error: {e}")

    # ── 26. Hash encode ──────────────────────────────────────────────────────

    @staticmethod
    def handle_hash_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Hash encoding for high-cardinality categorical columns."""
        col = params.get("column")
        n_features = params.get("n", 8)
        result = df.copy()
        cat_cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        created = []
        for c in cat_cols:
            for i in range(n_features):
                result[f"{c}_hash{i}"] = result[c].astype(str).apply(
                    lambda x, _i=i: (hash(x + str(_i)) % 2)
                )
                created.append(f"{c}_hash{i}")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Hash-encoded {len(cat_cols)} columns into {n_features} features each ({len(created)} total)",
        )

    # ── 27. Ordinal encode ───────────────────────────────────────────────────

    @staticmethod
    def handle_ordinal_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Ordinal encoding with auto-detected order (sorted unique values)."""
        col = params.get("column")
        result = df.copy()
        cat_cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include=["object", "category"]).columns.tolist()
        )
        encoded = []
        for c in cat_cols:
            sorted_vals = sorted(result[c].dropna().unique())
            mapping = {v: i for i, v in enumerate(sorted_vals)}
            result[f"{c}_ordinal"] = result[c].map(mapping)
            encoded.append(c)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Ordinal-encoded {len(encoded)} columns (alphabetical order)",
        )

    # ── 28. Label binarize ───────────────────────────────────────────────────

    @staticmethod
    def handle_label_binarize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Multi-label binarization — one binary column per unique value."""
        col = params.get("column")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        try:
            from sklearn.preprocessing import LabelBinarizer
            lb = LabelBinarizer()
            vals = result[col].fillna("__MISSING__")
            binarized = lb.fit_transform(vals)
            classes = lb.classes_
            for i, cls in enumerate(classes):
                result[f"{col}_{cls}"] = binarized[:, i] if binarized.ndim > 1 else binarized.ravel()
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Binarized '{col}' into {len(classes)} binary columns",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Label binarize error: {e}")

    # ── 29. Exponential transform ────────────────────────────────────────────

    @staticmethod
    def handle_exponential_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply exponential transform (e^x) to numeric columns."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            result[f"{col}_exp"] = np.exp(result[col].clip(upper=700))
            summary = f"Exponential transform of '{col}' → '{col}_exp'"
        else:
            transformed = []
            for c in result.select_dtypes(include="number").columns:
                if result[c].max() <= 700:
                    result[f"{c}_exp"] = np.exp(result[c])
                    transformed.append(c)
            summary = f"Exponential-transformed {len(transformed)} columns (max<=700)"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 30. Sin/cos hour encoding ────────────────────────────────────────────

    @staticmethod
    def handle_sin_cos_hour(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Sin/cos encoding for hour-of-day column (period=24)."""
        col = params.get("column")
        result = df.copy()
        if col and col in result.columns:
            hours = pd.to_numeric(result[col], errors="coerce")
        else:
            dt_cols = result.select_dtypes(include="datetime").columns.tolist()
            if not dt_cols:
                return HandlerResult(success=False, error="No datetime or hour column found")
            col = dt_cols[0]
            hours = result[col].dt.hour
        result[f"{col}_hour_sin"] = np.sin(2 * np.pi * hours / 24).round(4)
        result[f"{col}_hour_cos"] = np.cos(2 * np.pi * hours / 24).round(4)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Sin/cos hour encoding from '{col}' (period=24)",
        )

    # ── 31. Is weekend ───────────────────────────────────────────────────────

    @staticmethod
    def handle_is_weekend(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create binary weekend flag from datetime column."""
        col = params.get("column")
        result = df.copy()
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
            return HandlerResult(success=False, error="No datetime column found")
        for c in dt_cols:
            result[f"{c}_is_weekend"] = result[c].dt.dayofweek.isin([5, 6]).astype(int)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created is_weekend flag for {len(dt_cols)} datetime column(s)",
        )

    # ── 32. Is holiday ───────────────────────────────────────────────────────

    @staticmethod
    def handle_is_holiday(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Basic holiday detection — weekends + common fixed-date holidays."""
        col = params.get("column")
        result = df.copy()
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
            return HandlerResult(success=False, error="No datetime column found")
        common_holidays = {(1, 1), (7, 4), (12, 25), (12, 31), (1, 26), (5, 1)}
        for c in dt_cols:
            is_wknd = result[c].dt.dayofweek.isin([5, 6])
            month_day = list(zip(result[c].dt.month, result[c].dt.day))
            is_fixed = pd.Series([md in common_holidays for md in month_day], index=result.index)
            result[f"{c}_is_holiday"] = (is_wknd | is_fixed).astype(int)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created is_holiday flag for {len(dt_cols)} datetime column(s)",
        )

    # ── 33. Time since ───────────────────────────────────────────────────────

    @staticmethod
    def handle_time_since(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Days since a reference date (default: first date in column)."""
        col = params.get("column")
        ref_date = params.get("reference")
        result = df.copy()
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
            return HandlerResult(success=False, error="No datetime column found")
        for c in dt_cols:
            if ref_date:
                ref = pd.to_datetime(ref_date)
            else:
                ref = result[c].min()
            result[f"{c}_days_since"] = (result[c] - ref).dt.days
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created days_since features for {len(dt_cols)} column(s) (ref={ref_date or 'first date'})",
        )

    # ── 34. Distance from mean ───────────────────────────────────────────────

    @staticmethod
    def handle_distance_from_mean(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute distance from column mean or median."""
        col = params.get("column")
        method = params.get("method", "mean")
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        created = []
        for c in cols:
            center = result[c].mean() if method == "mean" else result[c].median()
            result[f"{c}_dist_{method}"] = (result[c] - center).round(4)
            created.append(f"{c}_dist_{method}")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} distance-from-{method} features",
        )

    # ── 35. Clip features ────────────────────────────────────────────────────

    @staticmethod
    def handle_clip_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Clip numeric values to percentile range (default 1st-99th)."""
        col = params.get("column")
        lower_pct = params.get("lower", 0.01)
        upper_pct = params.get("upper", 0.99)
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        clipped = []
        for c in cols:
            lo = result[c].quantile(lower_pct)
            hi = result[c].quantile(upper_pct)
            result[c] = result[c].clip(lo, hi)
            clipped.append(c)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Clipped {len(clipped)} columns to [{lower_pct:.0%}, {upper_pct:.0%}] percentile range",
        )

    # ── 36. Auto feature select ──────────────────────────────────────────────

    @staticmethod
    def handle_auto_feature_select(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Auto-select features: drop low-variance + highly-correlated columns."""
        var_threshold = params.get("var_threshold", 0.01)
        corr_threshold = params.get("corr_threshold", 0.95)
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        dropped = []
        # Low variance
        variances = result[num_cols].var()
        low_var = variances[variances < var_threshold].index.tolist()
        result = result.drop(columns=low_var)
        dropped.extend(low_var)
        # High correlation
        remaining = [c for c in num_cols if c not in low_var]
        if len(remaining) >= 2:
            corr = result[remaining].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
            high_corr = [c for c in upper.columns if upper[c].max() > corr_threshold]
            result = result.drop(columns=high_corr)
            dropped.extend(high_corr)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Auto-selected features: dropped {len(dropped)} columns (low-var: {low_var}, high-corr: {high_corr if len(remaining) >= 2 else []})",
        )

    # ── 37. Feature cross ────────────────────────────────────────────────────

    @staticmethod
    def handle_feature_cross(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cross two categorical columns to create A_x_B combination feature."""
        cols = params.get("columns", [])
        result = df.copy()
        cat_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(cols) >= 2:
            cols = cols[:2]
        elif len(cat_cols) >= 2:
            cols = cat_cols[:2]
        else:
            return HandlerResult(success=False, error="Need at least 2 categorical columns")
        name = f"{cols[0]}_x_{cols[1]}"
        result[name] = result[cols[0]].astype(str) + "_x_" + result[cols[1]].astype(str)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created feature cross '{name}' ({result[name].nunique()} unique combinations)",
        )

    # ── 38. Rolling stats features ───────────────────────────────────────────

    @staticmethod
    def handle_rolling_stats_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Add rolling mean/std/min/max as new feature columns."""
        col = params.get("column")
        window = params.get("window", 3)
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()[:3]
        )
        created = []
        for c in cols:
            for agg in ["mean", "std", "min", "max"]:
                name = f"{c}_roll{window}_{agg}"
                result[name] = result[c].rolling(window, min_periods=1).agg(agg).round(4)
                created.append(name)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} rolling stats features (window={window})",
        )

    # ── 39. EWM features ─────────────────────────────────────────────────────

    @staticmethod
    def handle_ewm_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Exponentially weighted moving average/std as features."""
        col = params.get("column")
        span = params.get("span", 5)
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()[:3]
        )
        created = []
        for c in cols:
            result[f"{c}_ewm_mean"] = result[c].ewm(span=span, min_periods=1).mean().round(4)
            result[f"{c}_ewm_std"] = result[c].ewm(span=span, min_periods=1).std().round(4)
            created.extend([f"{c}_ewm_mean", f"{c}_ewm_std"])
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} EWM features (span={span})",
        )

    # ── 40. Target binary encode ─────────────────────────────────────────────

    @staticmethod
    def handle_target_binary_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode a numeric target as binary (1 if above median, 0 otherwise)."""
        col = params.get("column")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not col or col not in df.columns:
            col = num_cols[-1] if num_cols else None
        if col is None:
            return HandlerResult(success=False, error="No target column found")
        result = df.copy()
        median_val = result[col].median()
        result[f"{col}_binary"] = (result[col] >= median_val).astype(int)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Binary-encoded '{col}' (1 if >= median {median_val:.2f}, else 0)",
        )

    # ── 41. Z-score features ─────────────────────────────────────────────────

    @staticmethod
    def handle_zscore_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Add z-score columns for numeric features."""
        col = params.get("column")
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        created = []
        for c in cols:
            mean = result[c].mean()
            std = result[c].std()
            if std > 0:
                result[f"{c}_zscore"] = ((result[c] - mean) / std).round(4)
                created.append(f"{c}_zscore")
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Created {len(created)} z-score feature columns",
        )

    # ── 42. Box-Cox transform ────────────────────────────────────────────────

    @staticmethod
    def handle_boxcox_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Box-Cox transform — requires strictly positive values."""
        col = params.get("column")
        result = df.copy()
        try:
            from scipy.stats import boxcox
            cols = (
                [col] if col and col in result.columns
                else result.select_dtypes(include="number").columns.tolist()
            )
            transformed = []
            for c in cols:
                vals = result[c].dropna()
                if (vals > 0).all() and len(vals) > 1:
                    t_vals, lam = boxcox(vals.values)
                    result[f"{c}_boxcox"] = np.nan
                    result.loc[vals.index, f"{c}_boxcox"] = np.round(t_vals, 4)
                    transformed.append(f"{c} (lambda={lam:.3f})")
            if not transformed:
                return HandlerResult(success=False, error="No columns with all-positive values found")
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Box-Cox transformed {len(transformed)} columns: {transformed}",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Box-Cox error: {e}")

    # ── 43. Yeo-Johnson transform ────────────────────────────────────────────

    @staticmethod
    def handle_yeo_johnson_transform(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Yeo-Johnson transform — handles negative values unlike Box-Cox."""
        col = params.get("column")
        result = df.copy()
        try:
            from sklearn.preprocessing import PowerTransformer
            cols = (
                [col] if col and col in result.columns
                else result.select_dtypes(include="number").columns.tolist()
            )
            X = result[cols].dropna()
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            transformed = pd.DataFrame(pt.fit_transform(X), index=X.index, columns=cols)
            for c in cols:
                result[f"{c}_yeojohnson"] = np.nan
                result.loc[X.index, f"{c}_yeojohnson"] = transformed[c].round(4)
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Yeo-Johnson transformed {len(cols)} columns",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Yeo-Johnson error: {e}")

    # ── 44. Winsorize ────────────────────────────────────────────────────────

    @staticmethod
    def handle_winsorize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cap extreme values at percentile bounds (default 5th-95th)."""
        col = params.get("column")
        lower_pct = params.get("lower", 0.05)
        upper_pct = params.get("upper", 0.95)
        result = df.copy()
        cols = (
            [col] if col and col in result.columns
            else result.select_dtypes(include="number").columns.tolist()
        )
        winsorized = []
        for c in cols:
            lo = result[c].quantile(lower_pct)
            hi = result[c].quantile(upper_pct)
            result[c] = result[c].clip(lo, hi)
            winsorized.append(c)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Winsorized {len(winsorized)} columns to [{lower_pct:.0%}, {upper_pct:.0%}]",
        )

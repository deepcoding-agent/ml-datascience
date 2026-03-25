"""Transform handler — 50 handlers: filter, sort, groupby, assign, encode, scale, pivot, melt, rank, rolling, merge, resample, etc."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from api.handlers.base import BaseHandler, HandlerResult
from api.logger import get_logger

log = get_logger(__name__)


class TransformHandler(BaseHandler):

    @staticmethod
    def handle_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        op = params.get("operator", "==")
        val = params.get("value")
        if val is None:
            return HandlerResult(success=False, error="No filter value provided")
        ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
        method = ops.get(op, "eq")
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                val = float(val)
            result = df[getattr(df[col], method)(val)]
        except Exception as e:
            return HandlerResult(success=False, error=f"Filter error: {e}")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Filtered {col} {op} {val}: {len(df):,} → {len(result):,} rows")

    @staticmethod
    def handle_assign_value(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        val = params.get("value")
        if val is None:
            return HandlerResult(success=False, error="No value to assign")
        result = df.copy()
        result[col] = val
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Set all {len(result):,} rows of '{col}' to {val}")

    @staticmethod
    def handle_sort(df: pd.DataFrame, params: dict) -> HandlerResult:
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        ascending = params.get("ascending", True)
        result = df.sort_values(col, ascending=ascending).reset_index(drop=True)
        order = "ascending" if ascending else "descending"
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Sorted by '{col}' {order}")

    @staticmethod
    def handle_groupby_agg(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        agg = params.get("agg", "count")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        if agg == "count":
            result = df.groupby(col, dropna=False).size().reset_index(name="count")
        else:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            if not num_cols:
                return HandlerResult(success=False, error="No numeric columns to aggregate")
            result = df.groupby(col, dropna=False)[num_cols].agg(agg).reset_index()
        result = result.sort_values(result.columns[-1], ascending=False).reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Grouped by '{col}' with {agg}")

    @staticmethod
    def handle_add_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        expression = params.get("expression", "")
        if not col:
            return HandlerResult(success=False, error="No column name provided")
        result = df.copy()
        try:
            result[col] = result.eval(expression) if expression else 0
        except Exception as e:
            return HandlerResult(success=False, error=f"Expression error: {e}")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Added column '{col}'")

    @staticmethod
    def handle_encode_label(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        mappings = {}
        for c in cols:
            le = LabelEncoder()
            result[c] = le.fit_transform(result[c].astype(str))
            mappings[c] = {str(cls): int(i) for i, cls in enumerate(le.classes_)}
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Label-encoded {len(cols)} columns", metadata={"mappings": mappings})

    @staticmethod
    def handle_encode_onehot(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        result = df.copy()
        cols = [col] if col and col in result.columns else result.select_dtypes(include=["object", "category"]).columns.tolist()
        result = pd.get_dummies(result, columns=cols, drop_first=True, dtype=int)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"One-hot encoded {len(cols)} columns → {result.shape[1]} total cols")

    @staticmethod
    def handle_scale_minmax(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = MinMaxScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"MinMax scaled {len(num_cols)} numeric columns to [0,1]")

    @staticmethod
    def handle_scale_standard(df: pd.DataFrame, params: dict) -> HandlerResult:
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = StandardScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Standard scaled {len(num_cols)} numeric columns (mean=0, std=1)")

    @staticmethod
    def handle_bin_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        col = params.get("column")
        bins = params.get("n", 5)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        result[f"{col}_bin"] = pd.cut(result[col], bins=bins, labels=False)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Binned '{col}' into {bins} bins → '{col}_bin'")

    @staticmethod
    def handle_inject_null(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Inject random NaN values into a copy of the DataFrame."""
        fraction = params.get("value", 15)
        if fraction > 1:
            fraction = fraction / 100.0  # convert 15 → 0.15
        result = df.copy()
        for col in result.columns:
            n_nulls = int(len(result) * fraction)
            if n_nulls > 0:
                null_indices = np.random.choice(result.index, size=n_nulls, replace=False)
                result.loc[null_indices, col] = np.nan
        total_nulls = int(result.isnull().sum().sum())
        total_cells = result.shape[0] * result.shape[1]
        actual_pct = total_nulls / total_cells * 100 if total_cells > 0 else 0
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Injected ~{fraction*100:.0f}% null values. Actual: {actual_pct:.1f}% ({total_nulls:,}/{total_cells:,} cells)",
        )

    @staticmethod
    def handle_sample_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = min(params.get("n", 10), len(df))
        result = df.sample(n=n, random_state=42)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Random sample of {n} rows")

    @staticmethod
    def handle_head(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = params.get("n", 10)
        result = df.head(n)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"First {n} rows")

    @staticmethod
    def handle_tail(df: pd.DataFrame, params: dict) -> HandlerResult:
        n = params.get("n", 10)
        result = df.tail(n)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Last {n} rows")

    # ── 1. Pivot table ───────────────────────────────────────────────────────

    @staticmethod
    def handle_pivot(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Pivot table: rows=index, columns=columns, values=values, aggfunc=agg."""
        index = params.get("index") or params.get("column")
        columns = params.get("columns")
        values = params.get("values")
        agg = params.get("agg", "mean")

        if not index or index not in df.columns:
            return HandlerResult(success=False, error=f"Index column '{index}' not found")

        try:
            if columns and columns in df.columns and values and values in df.columns:
                result = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=agg)
                result = result.reset_index()
                result.columns = [str(c) for c in result.columns]
            elif values and values in df.columns:
                result = pd.pivot_table(df, index=index, values=values, aggfunc=agg).reset_index()
            else:
                num_cols = df.select_dtypes(include="number").columns.tolist()
                result = pd.pivot_table(df, index=index, values=num_cols, aggfunc=agg).reset_index()

            return HandlerResult(success=True, result_df=result, output_type="query",
                                 summary=f"Pivot table: index='{index}', agg={agg} ({result.shape[0]}×{result.shape[1]})")
        except Exception as e:
            return HandlerResult(success=False, error=f"Pivot error: {e}")

    # ── 2. Melt (wide → long) ────────────────────────────────────────────────

    @staticmethod
    def handle_melt(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Unpivot/melt: wide → long format."""
        id_vars = params.get("id_vars", [])
        value_vars = params.get("value_vars", [])

        if not id_vars:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            id_vars = cat_cols[:2] if cat_cols else [df.columns[0]]
        id_vars = [c for c in id_vars if c in df.columns]

        if not value_vars:
            value_vars = [c for c in df.columns if c not in id_vars]
        value_vars = [c for c in value_vars if c in df.columns]

        result = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                         var_name="variable", value_name="value")
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Melted: {len(value_vars)} columns → long format ({len(result):,} rows)")

    # ── 3. Robust scaler ─────────────────────────────────────────────────────

    @staticmethod
    def handle_scale_robust(df: pd.DataFrame, params: dict) -> HandlerResult:
        """RobustScaler — scales using median/IQR, resistant to outliers."""
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to scale")
        result[num_cols] = RobustScaler().fit_transform(result[num_cols])
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Robust scaled {len(num_cols)} numeric columns (median-centered, IQR-scaled)")

    # ── 4. N largest ──────────────────────────────────────────────────────────

    @staticmethod
    def handle_nlargest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Top N rows by column value."""
        col = params.get("column")
        n = params.get("n", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        result = df.nlargest(min(n, len(df)), col)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Top {min(n, len(df))} rows by '{col}'")

    # ── 5. N smallest ────────────────────────────────────────────────────────

    @staticmethod
    def handle_nsmallest(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bottom N rows by column value."""
        col = params.get("column")
        n = params.get("n", 10)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else df.columns[0])
        result = df.nsmallest(min(n, len(df)), col)
        return HandlerResult(success=True, result_df=result, output_type="query",
                             summary=f"Bottom {min(n, len(df))} rows by '{col}'")

    # ── 6. Rank ──────────────────────────────────────────────────────────────

    @staticmethod
    def handle_rank(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Add rank column based on a numeric column."""
        col = params.get("column")
        ascending = params.get("ascending", True)
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column to rank")
        result = df.copy()
        result[f"{col}_rank"] = result[col].rank(ascending=ascending, method="min").astype(int)
        result = result.sort_values(f"{col}_rank").reset_index(drop=True)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Ranked by '{col}' {'ascending' if ascending else 'descending'}")

    # ── 7. Cumulative ────────────────────────────────────────────────────────

    @staticmethod
    def handle_cumulative(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cumulative sum, count, max, or min."""
        col = params.get("column")
        func = params.get("agg", "sum")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column for cumulative")
        result = df.copy()
        if func == "sum":
            result[f"{col}_cumsum"] = result[col].cumsum()
        elif func == "max":
            result[f"{col}_cummax"] = result[col].cummax()
        elif func == "min":
            result[f"{col}_cummin"] = result[col].cummin()
        elif func == "count":
            result[f"{col}_cumcount"] = range(1, len(result) + 1)
        else:
            result[f"{col}_cumsum"] = result[col].cumsum()
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Cumulative {func} of '{col}'")

    # ── 8. Rolling window ────────────────────────────────────────────────────

    @staticmethod
    def handle_rolling(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Rolling/moving window: mean, sum, std, min, max."""
        col = params.get("column")
        window = params.get("window", 3)
        func = params.get("agg", "mean")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        col = col if col and col in df.columns else (num_cols[0] if num_cols else None)
        if not col:
            return HandlerResult(success=False, error="No column for rolling")
        result = df.copy()
        roller = result[col].rolling(window=window, min_periods=1)
        result[f"{col}_rolling_{func}_{window}"] = getattr(roller, func)().round(4)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Rolling {func} (window={window}) of '{col}'")

    # ── 9. Round values ──────────────────────────────────────────────────────

    @staticmethod
    def handle_round_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Round numeric columns to N decimal places."""
        col = params.get("column")
        decimals = params.get("decimals", 2)
        result = df.copy()
        if col and col in result.columns:
            result[col] = result[col].round(decimals)
            summary = f"Rounded '{col}' to {decimals} decimals"
        else:
            num_cols = result.select_dtypes(include="number").columns.tolist()
            result[num_cols] = result[num_cols].round(decimals)
            summary = f"Rounded {len(num_cols)} numeric columns to {decimals} decimals"
        return HandlerResult(success=True, result_df=result, output_type="generate", summary=summary)

    # ── 10. Split column ─────────────────────────────────────────────────────

    @staticmethod
    def handle_split_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split a string column by delimiter into multiple columns."""
        col = params.get("column")
        delimiter = params.get("delimiter", ",")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        split_df = result[col].astype(str).str.split(delimiter, expand=True)
        split_df.columns = [f"{col}_{i+1}" for i in range(split_df.shape[1])]
        for c in split_df.columns:
            split_df[c] = split_df[c].str.strip()
        result = pd.concat([result, split_df], axis=1)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Split '{col}' by '{delimiter}' → {split_df.shape[1]} new columns")

    # ── 11. Concat columns ───────────────────────────────────────────────────

    @staticmethod
    def handle_concat_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Concatenate string columns into a new column."""
        cols = params.get("columns", [])
        separator = params.get("separator", "_")
        new_name = params.get("new_name", "combined")
        cols = [c for c in cols if c in df.columns]
        if len(cols) < 2:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            cols = cat_cols[:2] if len(cat_cols) >= 2 else df.columns[:2].tolist()
        result = df.copy()
        result[new_name] = result[cols[0]].astype(str)
        for c in cols[1:]:
            result[new_name] = result[new_name] + separator + result[c].astype(str)
        return HandlerResult(success=True, result_df=result, output_type="generate",
                             summary=f"Concatenated {cols} → '{new_name}'")

    # ── 12. Quantile-based binning ───────────────────────────────────────────

    @staticmethod
    def handle_qcut(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Quantile-based binning (equal-frequency bins)."""
        col = params.get("column")
        q = params.get("n", 4)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        result = df.copy()
        try:
            result[f"{col}_qbin"] = pd.qcut(result[col], q=q, labels=False, duplicates="drop")
            actual_bins = result[f"{col}_qbin"].nunique()
            return HandlerResult(success=True, result_df=result, output_type="generate",
                                 summary=f"Quantile-binned '{col}' into {actual_bins} bins → '{col}_qbin'")
        except Exception as e:
            return HandlerResult(success=False, error=f"Quantile binning error: {e}")

    # ── 13. Merge / Join two datasets ────────────────────────────────────

    @staticmethod
    def handle_merge(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Merge (join) current DataFrame with itself via self-join on a key,
        or create a cross-tabulated version. Use how=inner/left/right/outer."""
        column = params.get("column")
        how = params.get("how", "inner")
        if not column or column not in df.columns:
            return HandlerResult(success=False, error=f"Column '{column}' not found for merge key")
        # Self-join dedup: group by key → aggregate all numeric columns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        agg_dict = {c: "sum" for c in num_cols if c != column}
        if not agg_dict:
            agg_dict = {df.columns[0]: "count"}
        result = df.groupby(column, dropna=False).agg(agg_dict).reset_index()
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Merged by '{column}' ({how}): {len(df):,} → {len(result):,} rows",
        )

    # ── 14. Transpose ────────────────────────────────────────────────────

    @staticmethod
    def handle_transpose(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Transpose the DataFrame (swap rows and columns)."""
        result = df.T.reset_index()
        result.columns = ["feature"] + [f"row_{i}" for i in range(len(result.columns) - 1)]
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Transposed: {df.shape[0]}×{df.shape[1]} → {result.shape[0]}×{result.shape[1]}",
        )

    # ── 15. Drop rows by index range ─────────────────────────────────────

    @staticmethod
    def handle_drop_rows(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Drop rows by index range or specific indices."""
        start = params.get("start")
        end = params.get("end")
        indices = params.get("indices")  # list of int
        result = df.copy()
        original = len(result)

        if indices and isinstance(indices, list):
            result = result.drop(index=[i for i in indices if i in result.index]).reset_index(drop=True)
        elif start is not None and end is not None:
            result = result.drop(index=range(int(start), int(end) + 1), errors="ignore").reset_index(drop=True)
        elif start is not None:
            result = result.iloc[int(start):].reset_index(drop=True)
        else:
            return HandlerResult(success=False, error="Specify start/end range or indices list")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Dropped rows: {original:,} → {len(result):,} ({original - len(result)} removed)",
        )

    # ── 16. Shuffle rows ────────────────────────────────────────────────

    @staticmethod
    def handle_shuffle(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Randomly shuffle all rows. Useful before train/test splits."""
        seed = int(params.get("seed", 42))
        result = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Shuffled {len(result):,} rows (seed={seed})",
        )

    # ── 17. Train/test split ─────────────────────────────────────────────

    @staticmethod
    def handle_train_test_split(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split dataset into train and test sets. Adds a _split column (train/test)."""
        test_size = float(params.get("test_size", 0.2))
        seed = int(params.get("seed", 42))
        stratify_col = params.get("column")

        result = df.copy()
        n_test = max(1, int(len(result) * test_size))
        n_train = len(result) - n_test

        if stratify_col and stratify_col in result.columns:
            try:
                from sklearn.model_selection import train_test_split as sk_split
                train_idx, test_idx = sk_split(
                    result.index, test_size=test_size, random_state=seed,
                    stratify=result[stratify_col],
                )
                result.loc[train_idx, "_split"] = "train"
                result.loc[test_idx, "_split"] = "test"
            except Exception:
                rng = np.random.RandomState(seed)
                mask = rng.rand(len(result)) >= test_size
                result["_split"] = np.where(mask, "train", "test")
        else:
            rng = np.random.RandomState(seed)
            mask = rng.rand(len(result)) >= test_size
            result["_split"] = np.where(mask, "train", "test")

        train_n = int((result["_split"] == "train").sum())
        test_n = int((result["_split"] == "test").sum())
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Split: {train_n} train + {test_n} test ({test_size*100:.0f}% test)",
        )

    # ── 18. Clip values ──────────────────────────────────────────────────

    @staticmethod
    def handle_clip(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Clip numeric values to min/max bounds."""
        col = params.get("column")
        lower = params.get("min")
        upper = params.get("max")
        result = df.copy()

        if col and col in result.columns:
            cols = [col]
        else:
            cols = result.select_dtypes(include="number").columns.tolist()

        for c in cols:
            lo = float(lower) if lower is not None else None
            hi = float(upper) if upper is not None else None
            result[c] = result[c].clip(lower=lo, upper=hi)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Clipped {len(cols)} column(s) to [{lower}, {upper}]",
        )

    # ── 19. Replace values conditionally ─────────────────────────────────

    @staticmethod
    def handle_where(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Replace values where condition is NOT met (like np.where).
        Keeps values matching condition, replaces others."""
        col = params.get("column")
        operator = params.get("operator", ">")
        value = params.get("value")
        replacement = params.get("replacement", np.nan)
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")
        if value is None:
            return HandlerResult(success=False, error="Specify value= parameter")

        result = df.copy()
        ops = {"==": "eq", "!=": "ne", ">": "gt", "<": "lt", ">=": "ge", "<=": "le"}
        method = ops.get(operator, "gt")
        try:
            if pd.api.types.is_numeric_dtype(result[col]):
                value = float(value)
            mask = getattr(result[col], method)(value)
            result[col] = result[col].where(mask, other=replacement)
        except Exception as e:
            return HandlerResult(success=False, error=f"Where error: {e}")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Replaced '{col}' where NOT ({operator} {value}) → {replacement}",
        )

    # ── 20. Explode list column ──────────────────────────────────────────

    @staticmethod
    def handle_explode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Explode a column containing comma-separated values or lists into separate rows."""
        col = params.get("column")
        delimiter = params.get("delimiter", ",")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")

        result = df.copy()
        original = len(result)
        # Split string column into lists if needed
        if result[col].dtype == "object":
            result[col] = result[col].fillna("").astype(str).str.split(delimiter)
        result = result.explode(col, ignore_index=True)
        result[col] = result[col].astype(str).str.strip()

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Exploded '{col}': {original:,} → {len(result):,} rows",
        )

    # ── 21. Encode target (binary) ───────────────────────────────────────

    @staticmethod
    def handle_encode_binary(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode a column as binary (0/1) based on a threshold or specific value."""
        col = params.get("column")
        threshold = params.get("threshold")
        positive_value = params.get("value")
        if not col or col not in df.columns:
            return HandlerResult(success=False, error=f"Column '{col}' not found")

        result = df.copy()
        if threshold is not None:
            result[f"{col}_binary"] = (result[col] >= float(threshold)).astype(int)
            desc = f"'{col}' >= {threshold} → 1, else 0"
        elif positive_value is not None:
            result[f"{col}_binary"] = (result[col] == positive_value).astype(int)
            desc = f"'{col}' == {positive_value} → 1, else 0"
        else:
            # Auto: if 2 unique values, encode the more common as 1
            uniques = result[col].dropna().unique()
            if len(uniques) == 2:
                result[f"{col}_binary"] = (result[col] == uniques[0]).astype(int)
                desc = f"'{col}': {uniques[0]}=1, {uniques[1]}=0"
            else:
                return HandlerResult(success=False, error="Specify threshold or value for binary encoding")

        ones = int(result[f"{col}_binary"].sum())
        zeros = len(result) - ones
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Binary encoded: {desc} ({ones} positive, {zeros} negative)",
        )

    # ── 22. Percent change ───────────────────────────────────────────────

    @staticmethod
    def handle_pct_change(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute percentage change between consecutive rows."""
        col = params.get("column")
        periods = int(params.get("periods", 1))
        result = df.copy()

        if col and col in result.columns:
            cols = [col]
        else:
            cols = result.select_dtypes(include="number").columns.tolist()[:5]

        created: list[str] = []
        for c in cols:
            name = f"{c}_pct_change"
            result[name] = result[c].pct_change(periods=periods).round(4)
            created.append(name)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Percentage change (periods={periods}) for {len(created)} column(s)",
        )

    # ── 23. Normalize to percentage (per-row) ────────────────────────────

    @staticmethod
    def handle_normalize_pct(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Normalize numeric columns to percentages (row-wise or column-wise)."""
        axis = params.get("axis", "columns")  # columns (per-row) or index (per-column)
        result = df.copy()
        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns")

        if axis == "columns":
            # Each row sums to 100%
            row_sums = result[num_cols].sum(axis=1).replace(0, 1)
            for c in num_cols:
                result[c] = (result[c] / row_sums * 100).round(2)
            desc = "row-wise (each row sums to 100%)"
        else:
            # Each column sums to 100%
            for c in num_cols:
                total = result[c].sum()
                if total != 0:
                    result[c] = (result[c] / total * 100).round(2)
            desc = "column-wise (each column sums to 100%)"

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Normalized {len(num_cols)} columns to percentages ({desc})",
        )

    # ── 24. Apply math expression ────────────────────────────────────────

    @staticmethod
    def handle_apply_expr(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Apply a mathematical expression to create a new column.
        Expression uses column names as variables: 'price / area'."""
        expression = params.get("expression", "")
        new_name = params.get("new_name", "result")
        if not expression:
            return HandlerResult(success=False, error="Specify expression= parameter (e.g. 'price / area')")

        result = df.copy()
        try:
            result[new_name] = result.eval(expression)
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Created '{new_name}' = {expression}",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Expression error: {e}")

    # ── 25. Flatten multi-index / reset column names ─────────────────────

    @staticmethod
    def handle_flatten_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Flatten multi-level column names into single-level snake_case names.
        Also standardizes all column names to lowercase snake_case."""
        result = df.copy()
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ["_".join(str(c) for c in col if str(c) != "").strip("_")
                              for col in result.columns]

        import re as _re
        new_names = {}
        for c in result.columns:
            name = str(c).strip()
            name = _re.sub(r"[^\w\s]", "", name)
            name = _re.sub(r"\s+", "_", name)
            name = name.lower().strip("_")
            if not name:
                name = f"col_{list(result.columns).index(c)}"
            new_names[c] = name
        result = result.rename(columns=new_names)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Flattened {len(result.columns)} columns to snake_case",
        )

    # ── 26. Resample time series ─────────────────────────────────────────

    @staticmethod
    def handle_resample(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Resample time series data to a different frequency (daily, weekly, monthly)."""
        date_col = params.get("column")
        freq = params.get("freq", "M")  # D, W, M, Q, Y
        agg = params.get("agg", "mean")

        # Find datetime column
        dt_cols = df.select_dtypes(include="datetime").columns.tolist()
        if date_col and date_col in df.columns:
            target = date_col
        elif dt_cols:
            target = dt_cols[0]
        else:
            # Try to parse object columns as dates
            for c in df.select_dtypes(include="object").columns:
                try:
                    df[c] = pd.to_datetime(df[c], format="mixed")
                    target = c
                    break
                except Exception:
                    continue
            else:
                return HandlerResult(success=False, error="No datetime column found for resampling")

        result = df.copy()
        result[target] = pd.to_datetime(result[target], format="mixed", errors="coerce")
        result = result.set_index(target)

        num_cols = result.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return HandlerResult(success=False, error="No numeric columns to aggregate")

        freq_map = {"daily": "D", "weekly": "W", "monthly": "ME", "quarterly": "QE", "yearly": "YE",
                    "M": "ME", "Q": "QE", "Y": "YE"}
        freq = freq_map.get(freq.lower(), freq)

        try:
            resampled = result[num_cols].resample(freq).agg(agg).reset_index()
            return HandlerResult(
                success=True, result_df=resampled, output_type="generate",
                summary=f"Resampled by '{target}' to {freq} ({agg}): {len(df):,} → {len(resampled):,} rows",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Resample error: {e}")

    # ── 27. Cross join / cartesian product ───────────────────────────────

    @staticmethod
    def handle_cross_join(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create cartesian product of unique values from two columns.
        Useful for generating all possible combinations."""
        columns = params.get("columns", [])
        if len(columns) < 2:
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            columns = cat_cols[:2] if len(cat_cols) >= 2 else df.columns[:2].tolist()

        c1, c2 = columns[0], columns[1]
        if c1 not in df.columns or c2 not in df.columns:
            return HandlerResult(success=False, error=f"Columns {c1}, {c2} not found")

        vals1 = df[c1].dropna().unique()
        vals2 = df[c2].dropna().unique()
        import itertools
        combos = list(itertools.product(vals1, vals2))
        result = pd.DataFrame(combos, columns=[c1, c2])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cross join: {len(vals1)} × {len(vals2)} = {len(result)} combinations",
        )

    # ── 28. Ordinal encoding with custom order ──────────────────────────

    @staticmethod
    def handle_encode_ordinal(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Ordinal-encode a categorical column using a custom order list.
        If no order is given, sorts unique values alphabetically."""
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        order = params.get("order")  # list of strings
        result = df.copy()

        if order and isinstance(order, list):
            mapping = {v: i for i, v in enumerate(order)}
        else:
            uniques = sorted(result[col].dropna().unique(), key=str)
            mapping = {v: i for i, v in enumerate(uniques)}

        result[f"{col}_ordinal"] = result[col].map(mapping)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Ordinal-encoded '{col}' → '{col}_ordinal' ({len(mapping)} levels)",
        )

    # ── 29. Shift column values ─────────────────────────────────────────

    @staticmethod
    def handle_shift_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Shift column values up or down by N rows (like lag/lead)."""
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        periods = int(params.get("periods", 1))
        result = df.copy()
        result[f"{col}_shift_{periods}"] = result[col].shift(periods)
        direction = "down" if periods > 0 else "up"
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Shifted '{col}' {direction} by {abs(periods)} rows → '{col}_shift_{periods}'",
        )

    # ── 30. Winsorize ───────────────────────────────────────────────────

    @staticmethod
    def handle_winsorize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Cap extreme values at percentile bounds (e.g. 5th/95th)."""
        col = params.get("column")
        lower = float(params.get("lower", 0.05))
        upper = float(params.get("upper", 0.95))
        result = df.copy()

        if col and col in result.columns:
            cols = [col]
        else:
            cols = result.select_dtypes(include="number").columns.tolist()

        for c in cols:
            lo = result[c].quantile(lower)
            hi = result[c].quantile(upper)
            result[c] = result[c].clip(lower=lo, upper=hi)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Winsorized {len(cols)} column(s) at [{lower*100:.0f}th, {upper*100:.0f}th] percentile",
        )

    # ── 31. Stack columns to long format ────────────────────────────────

    @staticmethod
    def handle_stack_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Stack selected columns into long format with col_name and col_value."""
        columns = params.get("columns", [])
        if not columns or not isinstance(columns, list):
            columns = df.columns.tolist()

        valid = [c for c in columns if c in df.columns]
        if not valid:
            return HandlerResult(success=False, error="No valid columns to stack")

        id_cols = [c for c in df.columns if c not in valid]
        result = df.melt(id_vars=id_cols if id_cols else None,
                         value_vars=valid,
                         var_name="col_name", value_name="col_value")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Stacked {len(valid)} columns → long format ({len(result):,} rows)",
        )

    # ── 32. Unstack column to wide format ───────────────────────────────

    @staticmethod
    def handle_unstack_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Unstack/pivot a column to wide format. Requires index, column, and value cols."""
        index_col = params.get("index")
        col_col = params.get("column")
        value_col = params.get("value")

        if not col_col or col_col not in df.columns:
            return HandlerResult(success=False, error="Specify column= for the column to unstack")

        if not index_col:
            candidates = [c for c in df.columns if c != col_col and c != value_col]
            index_col = candidates[0] if candidates else None
        if not value_col:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            value_col = num_cols[0] if num_cols else None

        if index_col is None or value_col is None:
            return HandlerResult(success=False, error="Need index, column, and value params")

        try:
            result = df.pivot_table(index=index_col, columns=col_col,
                                     values=value_col, aggfunc="first").reset_index()
            result.columns = [str(c) for c in result.columns]
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Unstacked '{col_col}' → {len(result.columns)-1} new columns, {len(result)} rows",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Unstack error: {e}")

    # ── 33. Reorder columns ─────────────────────────────────────────────

    @staticmethod
    def handle_reorder_columns(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Reorder columns alphabetically or by a provided list."""
        order = params.get("order")  # list of column names
        result = df.copy()

        if order and isinstance(order, list):
            valid = [c for c in order if c in result.columns]
            remaining = [c for c in result.columns if c not in valid]
            result = result[valid + remaining]
            desc = f"custom order ({len(valid)} specified)"
        else:
            result = result.reindex(sorted(result.columns), axis=1)
            desc = "alphabetical"

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Reordered {len(result.columns)} columns ({desc})",
        )

    # ── 34. Duplicate column ────────────────────────────────────────────

    @staticmethod
    def handle_duplicate_column(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Duplicate a column with a new name."""
        col, err = BaseHandler.require_column(df, params.get("column"), params.get("column", ""))
        if err:
            return err
        new_name = params.get("new_name", f"{col}_copy")
        result = df.copy()
        result[new_name] = result[col]
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Duplicated '{col}' → '{new_name}'",
        )

    # ── 35. Forward fill nulls ──────────────────────────────────────────

    @staticmethod
    def handle_fill_forward(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Forward-fill (ffill) null values only."""
        col = params.get("column")
        result = df.copy()

        if col and col in result.columns:
            before = int(result[col].isna().sum())
            result[col] = result[col].ffill()
            after = int(result[col].isna().sum())
            filled = before - after
            desc = f"Forward-filled '{col}': {filled} nulls filled"
        else:
            before = int(result.isna().sum().sum())
            result = result.ffill()
            after = int(result.isna().sum().sum())
            filled = before - after
            desc = f"Forward-filled all columns: {filled} nulls filled"

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=desc,
        )

    # ── 36. Interpolate missing values ──────────────────────────────────

    @staticmethod
    def handle_interpolate_values(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Interpolate missing numeric values using linear interpolation."""
        col = params.get("column")
        method = params.get("method", "linear")
        result = df.copy()

        if col and col in result.columns:
            cols = [col]
        else:
            cols = result.select_dtypes(include="number").columns.tolist()

        if not cols:
            return HandlerResult(success=False, error="No numeric columns to interpolate")

        before = int(result[cols].isna().sum().sum())
        for c in cols:
            result[c] = result[c].interpolate(method=method)
        after = int(result[cols].isna().sum().sum())
        filled = before - after

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Interpolated ({method}) {len(cols)} column(s): {filled} nulls filled",
        )

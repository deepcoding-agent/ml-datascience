"""handle_slope_chart handler."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.theme import _style
from api.logger import get_logger

log = get_logger(__name__)


def handle_slope_chart(df: pd.DataFrame, params: dict) -> HandlerResult:
    """Slope chart comparing two numeric columns (before/after or left/right)."""
    try:
        col_left = params.get("column")
        col_right = params.get("column2")
        label_col = params.get("label")

        num_cols = BaseHandler.get_numeric_cols(df)

        # Auto-select numeric columns if not specified
        if not col_left:
            if len(num_cols) >= 1:
                col_left = num_cols[0]
            else:
                return HandlerResult(
                    success=False,
                    error="No left/start column specified and no numeric columns found.",
                )
        if not col_right:
            if len(num_cols) >= 2:
                col_right = [c for c in num_cols if c != col_left][0]
            else:
                return HandlerResult(
                    success=False,
                    error="No right/end column specified and fewer than 2 numeric columns found.",
                )

        # Validate columns
        col_left, err = BaseHandler.require_column(df, col_left, col_left or "")
        if err:
            return err
        col_right, err = BaseHandler.require_column(df, col_right, col_right or "")
        if err:
            return err

        # Use label column or index
        if label_col and label_col in df.columns:
            labels = df[label_col].astype(str).tolist()
        else:
            labels = [f"Row {i}" for i in range(len(df))]

        # Cap rows for readability
        max_rows = min(len(df), 50)
        plot_df = df.head(max_rows).copy()
        plot_labels = labels[:max_rows]

        left_vals = plot_df[col_left].values
        right_vals = plot_df[col_right].values

        # Generate colors based on number of rows
        colorway = [
            "#FB8C3C", "#2EC4B6", "#457B9D", "#E71D36",
            "#FF9F1C", "#A8DADC", "#1D3557", "#6B4226",
        ]

        fig = go.Figure()

        for i in range(len(plot_df)):
            lv = left_vals[i]
            rv = right_vals[i]
            if pd.isna(lv) or pd.isna(rv):
                continue
            color = colorway[i % len(colorway)]
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[lv, rv],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=plot_labels[i],
                hovertemplate=(
                    f"<b>{plot_labels[i]}</b><br>"
                    f"{col_left}: %{{y:.2f}}<extra></extra>"
                ),
            ))

        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 1],
                ticktext=[col_left, col_right],
                range=[-0.2, 1.2],
            ),
        )

        _style(
            fig,
            title=f"Slope Chart: {col_left} vs {col_right} (n={len(plot_df)})",
            yaxis_title="Value",
            showlegend=len(plot_df) <= 20,
        )

        return HandlerResult(
            success=True,
            charts_plotly=[fig.to_json()],
            output_type="query",
            summary=(
                f"Slope chart comparing '{col_left}' (left) to '{col_right}' (right) "
                f"for {len(plot_df)} rows"
            ),
        )
    except Exception as e:
        log.exception("slope_chart failed")
        return HandlerResult(success=False, error=f"slope_chart failed: {e}")

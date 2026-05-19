"""POST /eda-report — generate a structured EDA report for a dataset."""
from __future__ import annotations

import base64
import io
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastapi import APIRouter

from api.logger import get_logger
from api.models import ColumnProfile, EDARequest, EDAResponse

router = APIRouter()
log = get_logger(__name__)

# Optional imports
try:
    import seaborn as sns
    _SNS = True
except ImportError:
    sns = None  # type: ignore[assignment]
    _SNS = False

try:
    import missingno as msno
    _MSNO = True
except ImportError:
    msno = None  # type: ignore[assignment]
    _MSNO = False


def _fig_to_b64() -> str:
    """Capture the current matplotlib figure as a base64 PNG string."""
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close("all")
    return b64


def _build_column_profiles(df: pd.DataFrame) -> list[ColumnProfile]:
    profiles: list[ColumnProfile] = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        profile = ColumnProfile(
            name=col,
            dtype=str(s.dtype),
            null_count=null_count,
            null_pct=round(null_count / len(df) * 100, 2) if len(df) > 0 else 0.0,
            unique_count=int(s.nunique()),
            top_values={str(k): int(v) for k, v in s.value_counts().head(5).items()},
        )
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe()
            profile.mean = round(float(desc.get("mean", 0)), 4)
            profile.std = round(float(desc.get("std", 0)), 4)
            profile.min = round(float(desc.get("min", 0)), 4)
            profile.max = round(float(desc.get("max", 0)), 4)
            profile.median = round(float(s.median()), 4)
            try:
                profile.skewness = round(float(s.skew()), 4)
            except Exception:
                pass
        profiles.append(profile)
    return profiles


def _generate_charts(df: pd.DataFrame) -> list[str]:
    charts: list[str] = []

    # 1. Distribution plots for numeric columns
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        n = min(len(num_cols), 12)
        cols_to_plot = num_cols[:n]
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(cols_to_plot):
            ax = axes_flat[i]
            data = df[col].dropna()
            if _SNS:
                sns.histplot(data, kde=True, ax=ax, color="#FB8C3C", edgecolor="white")
            else:
                ax.hist(data, bins=30, color="#FB8C3C", edgecolor="white")
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Distributions", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        charts.append(_fig_to_b64())

    # 2. Correlation heatmap
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        size = max(6, min(12, len(num_cols) * 0.8))
        plt.figure(figsize=(size, size * 0.8))
        if _SNS:
            sns.heatmap(
                corr, annot=len(num_cols) <= 15, fmt=".2f",
                cmap="RdYlBu_r", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
            )
        else:
            plt.imshow(corr.values, cmap="RdYlBu_r", aspect="auto")
            plt.colorbar(shrink=0.8)
            plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right", fontsize=8)
            plt.yticks(range(len(num_cols)), num_cols, fontsize=8)
        plt.title("Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        charts.append(_fig_to_b64())

    # 3. Missing value pattern
    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        if _MSNO:
            fig, ax = plt.subplots(figsize=(min(14, len(df.columns) * 0.6), 5))
            msno.matrix(df, ax=ax, sparkline=False, fontsize=8)
            ax.set_title("Missing Value Pattern", fontsize=13, fontweight="bold")
            plt.tight_layout()
            charts.append(_fig_to_b64())
        else:
            # Fallback: simple bar chart of missing values
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=True)
            if len(missing) > 0:
                plt.figure(figsize=(8, max(3, len(missing) * 0.4)))
                missing.plot.barh(color="#FB8C3C")
                plt.title("Missing Values per Column", fontsize=13, fontweight="bold")
                plt.xlabel("Count")
                plt.tight_layout()
                charts.append(_fig_to_b64())

    # 4. Box plots for numeric columns (outlier visualization)
    if num_cols:
        n = min(len(num_cols), 8)
        cols_to_plot = num_cols[:n]
        fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 4))
        axes_flat = [axes] if n == 1 else list(axes)
        for i, col in enumerate(cols_to_plot):
            if _SNS:
                sns.boxplot(y=df[col].dropna(), ax=axes_flat[i], color="#FB8C3C", width=0.5)
            else:
                axes_flat[i].boxplot(df[col].dropna(), patch_artist=True,
                                     boxprops=dict(facecolor="#FB8C3C"))
            axes_flat[i].set_title(col, fontsize=9, fontweight="bold")
            axes_flat[i].tick_params(labelsize=8)
        plt.suptitle("Box Plots (Outlier Detection)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        charts.append(_fig_to_b64())

    plt.close("all")
    return charts


@router.post("/eda-report", response_model=EDAResponse)
def eda_report(req: EDARequest) -> EDAResponse:
    t0 = time.perf_counter()
    log.info("▶ /eda-report  dataset='%s'  rows=%d", req.dataset.name, len(req.dataset.data))

    try:
        df = pd.DataFrame(req.dataset.data)

        # Basic info
        memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        dtypes_count = {str(k): int(v) for k, v in df.dtypes.value_counts().items()}

        # Column profiles
        profiles = _build_column_profiles(df)

        # Correlation matrix
        num_cols = df.select_dtypes(include="number").columns.tolist()
        corr_dict: dict = {}
        if len(num_cols) >= 2:
            corr = df[num_cols].corr().round(4)
            corr_dict = corr.to_dict()

        # Charts
        charts = _generate_charts(df)

        elapsed = round(time.perf_counter() - t0, 2)
        log.info(
            "✔ /eda-report done  cols=%d  charts=%d  elapsed=%.1fs",
            len(profiles), len(charts), elapsed,
        )

        return EDAResponse(
            rows=len(df),
            columns=len(df.columns),
            memory_mb=memory_mb,
            dtypes=dtypes_count,
            column_profiles=profiles,
            correlation=corr_dict,
            charts=charts,
            duration_seconds=elapsed,
        )

    except Exception as exc:
        log.exception("✘ /eda-report failed: %s", exc)
        return EDAResponse(
            rows=0, columns=0, memory_mb=0,
            duration_seconds=round(time.perf_counter() - t0, 2),
        )

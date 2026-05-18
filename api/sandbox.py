"""
Code sandbox — executes LLM-generated Python safely in an isolated namespace.

matplotlib is configured once at import time (Agg backend).
numpy is imported once at module level and injected into every sandbox run.

Hardening (Sprint 7):
- Import blocklist via overridden ``__builtins__["__import__"]`` (os, subprocess,
  socket, requests, etc. cannot be imported from user code).
- 30-second wall-clock timeout per ``run_code`` call via worker thread.
"""
from __future__ import annotations

import base64
import builtins
import io
import os
import threading
from contextlib import redirect_stdout
from typing import Any

# ── Limit joblib / sklearn to single-process so an aborted sandbox thread
# doesn't leave orphaned child processes pegging CPU on the host. With the
# daemon-thread "abandon" timeout below, multiprocessing workers can't be
# cleaned up properly — they keep running, accumulate temp dirs, and on
# Render they pile up until the worker is killed. Forcing threading +
# CPU count 1 makes timeouts safe and recoverable. Must be set before
# sklearn / joblib / loky import anywhere in the process.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from api.logger import get_logger

log = get_logger(__name__)

# ── Sandbox safety ────────────────────────────────────────────────────────────
# Modules user-generated code may NOT import. The library modules we pre-inject
# (pandas, numpy, plotly, seaborn, …) are unaffected because they go through the
# normal module loader at import time, before the sandbox namespace is built.
_BLOCKED_MODULES: frozenset[str] = frozenset({
    "os", "subprocess", "shutil", "pathlib", "sys",
    "socket", "ssl", "ctypes", "multiprocessing", "threading",
    "requests", "urllib", "urllib3", "http", "httpx",
    "ftplib", "smtplib", "telnetlib",
    "importlib", "pkgutil", "runpy",
})

_real_import = builtins.__import__


def _safe_import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """Reject imports of blocked roots; allow everything else."""
    root = name.split(".", 1)[0]
    if root in _BLOCKED_MODULES:
        raise ImportError(f"Sandbox: import of '{name}' is not allowed")
    return _real_import(name, globals, locals, fromlist, level)


_SANDBOX_BUILTINS: dict[str, Any] = {**vars(builtins), "__import__": _safe_import}

# Wall-clock timeout for a single run_code call.
_SANDBOX_TIMEOUT_SECONDS = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "30"))

# ── matplotlib — configure non-interactive backend + Thai font support ────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.font_manager as _fm

    # Find a font that supports Thai glyphs (fallback chain)
    _THAI_FONT = None
    for _candidate in ["Tahoma", "Arial Unicode MS", "Noto Sans Thai"]:
        try:
            _path = _fm.findfont(_fm.FontProperties(family=_candidate), fallback_to_default=False)
            if _path and "DejaVu" not in _path:
                _THAI_FONT = _candidate
                break
        except ValueError:
            pass
    if _THAI_FONT:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [_THAI_FONT, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False

    _MPL = True
except ImportError:
    _plt = None   # type: ignore[assignment]
    _MPL = False

# ── seaborn ──────────────────────────────────────────────────────────────────
try:
    import seaborn as _sns
    _SNS = True
except ImportError:
    _sns = None   # type: ignore[assignment]
    _SNS = False

# ── missingno ────────────────────────────────────────────────────────────────
try:
    import missingno as _msno
    _MSNO = True
except ImportError:
    _msno = None  # type: ignore[assignment]
    _MSNO = False

# ── plotly ───────────────────────────────────────────────────────────────────
try:
    import plotly
    import plotly.express as _px
    import plotly.graph_objects as _go
    import plotly.io as _pio
    import plotly.figure_factory as _ff
    from plotly.subplots import make_subplots as _make_subplots
    _PLOTLY = True
except ImportError:
    plotly = None  # type: ignore[assignment]
    _px = None     # type: ignore[assignment]
    _go = None     # type: ignore[assignment]
    _pio = None    # type: ignore[assignment]
    _ff = None     # type: ignore[assignment]
    _make_subplots = None  # type: ignore[assignment]
    _PLOTLY = False

# PrepPilot Plotly theme — applied to every captured figure
_PREPPILOT_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Noto Sans Thai, Tahoma, sans-serif", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter, Noto Sans Thai, Tahoma, sans-serif"),
    colorway=["#FF6B35", "#2EC4B6", "#E71D36", "#FF9F1C",
              "#6B4226", "#A8DADC", "#457B9D", "#1D3557"],
)


def _figure_has_data(fig) -> bool:  # type: ignore[no-untyped-def]
    """Return True iff at least one trace carries at least one data point.

    Drops figures the planner/codegen produced over text-heavy / empty data
    — those render as "Topic counts" axes with no bars. Per-trace check
    covers bar/scatter/line/box (x/y), pie/donut (values/labels), heatmap (z),
    maps (lat/lon), and finance (open/close). Anything else falls back to a
    truthy check on the first numeric-shaped attribute we can find.
    """
    if not fig.data:
        return False
    for trace in fig.data:
        for attr in ("x", "y", "values", "z", "labels", "lat", "lon", "open", "close"):
            v = getattr(trace, attr, None)
            if v is None:
                continue
            try:
                if len(v) > 0:
                    return True
            except TypeError:
                # Scalar attribute — treat as having data.
                return True
    return False

# ── Result variable priority (most explicit → least) ─────────────────────────
_RESULT_PRIORITY = [
    "result", "df_result", "output_df", "df_out",
    "df_new", "df_filtered", "filtered_df",
    "transformed_df", "df_clean", "df_transformed",
]


def run_code(
    code: str,
    df: pd.DataFrame,
    extra_dfs: dict[str, pd.DataFrame] | None = None,
) -> tuple[str, pd.DataFrame | None, str | None, pd.DataFrame | None, str | None]:
    """
    Execute *code* in a restricted namespace containing `df`, `pd`, `np`,
    `sns`, `msno`, `px`, `go`, and optionally extra named DataFrames.

    Wraps the exec in a worker thread with a 30s wall-clock timeout, and
    overrides ``__import__`` so user code cannot import blocked modules.

    Returns
    -------
    (stdout, result_df, chart_base64, sandbox_df, chart_json)
      stdout       — captured print output (or error message)
      result_df    — first new non-empty DataFrame detected in the namespace
      chart_base64 — first captured matplotlib figure as a PNG base64 string
      sandbox_df   — state of `df` after execution (may differ if mutated)
      chart_json   — first captured Plotly figure as a JSON string
    """
    # Default failure return — used if the worker thread doesn't return.
    timeout_result: tuple[str, None, None, None, None] = (
        f"Code execution timed out (>{_SANDBOX_TIMEOUT_SECONDS}s wall-clock limit)",
        None, None, None, None,
    )
    holder: list[tuple] = []

    def _worker() -> None:
        holder.append(_execute(code, df, extra_dfs))

    t = threading.Thread(target=_worker, daemon=True, name="sandbox-exec")
    t.start()
    t.join(timeout=_SANDBOX_TIMEOUT_SECONDS)

    if t.is_alive():
        log.warning("Sandbox exec exceeded %ds — abandoning thread", _SANDBOX_TIMEOUT_SECONDS)
        return timeout_result

    if not holder:
        return ("Code execution error: no result returned", None, None, None, None)

    return holder[0]


def _execute(
    code: str,
    df: pd.DataFrame,
    extra_dfs: dict[str, pd.DataFrame] | None,
) -> tuple[str, pd.DataFrame | None, str | None, pd.DataFrame | None, str | None]:
    """Run the user code inline; see ``run_code`` for the threaded wrapper."""
    buf = io.StringIO()
    captured_charts: list[str] = []
    captured_plotly: list[str] = []
    input_ids = {id(df)} | {id(v) for v in (extra_dfs or {}).values()}

    # Build sandbox namespace — numpy imported once at module level
    if _MPL:
        def _capture() -> None:
            fig = _plt.gcf()
            if fig.get_axes():
                img = io.BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight", dpi=130)
                img.seek(0)
                captured_charts.append(base64.b64encode(img.read()).decode())
                _plt.close(fig)

        original_show = _plt.show
        _plt.show = _capture  # type: ignore[method-assign]
        ns: dict[str, Any] = {"df": df, "pd": pd, "np": np, "plt": _plt}
    else:
        original_show = None
        ns = {"df": df, "pd": pd, "np": np}

    # Inject restricted __builtins__ so user code cannot import blocked modules
    ns["__builtins__"] = _SANDBOX_BUILTINS

    # Inject seaborn
    if _SNS:
        ns["sns"] = _sns
    # Inject missingno
    if _MSNO:
        ns["msno"] = _msno
    # Inject plotly
    if _PLOTLY:
        ns["px"] = _px
        ns["go"] = _go
        ns["ff"] = _ff
        ns["make_subplots"] = _make_subplots
        ns["plotly"] = plotly

    if extra_dfs:
        ns.update(extra_dfs)

    try:
        with redirect_stdout(buf):
            exec(compile(code, "<agent_code>", "exec"), ns)

        # Capture any figures created without an explicit plt.show()
        if _MPL:
            for _ in _plt.get_fignums():
                _capture()
            _plt.close("all")

        stdout = buf.getvalue().strip()

        # Try to render the last expression inline (e.g. bare `df.head()`)
        lines = [
            ln for ln in code.strip().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if lines:
            try:
                val = eval(compile(lines[-1], "<last>", "eval"), ns)
                if val is not None:
                    if isinstance(val, pd.DataFrame):
                        extra = val.head(5).to_markdown(index=False)
                    elif isinstance(val, pd.Series):
                        extra = val.to_string()
                    else:
                        extra = str(val)
                    stdout = (stdout + "\n" + extra).strip() if stdout else extra
            except Exception:
                pass

        stdout = stdout or "(code ran successfully, no output)"

        # Capture Plotly figures from namespace and apply PrepPilot theme.
        # Empty figures (axes only, no actual data) are silently dropped — they
        # show up as misleading "blank chart" cards in the UI when the codegen
        # ran on text-heavy / unique-per-row data the planner shouldn't have
        # picked. See _figure_has_data() above and the planner data-suitability
        # rules in api/agents/planner.py.
        if _PLOTLY:
            seen_ids: set[int] = set()
            for val in ns.values():
                if hasattr(val, "to_json") and type(val).__module__.startswith("plotly") and id(val) not in seen_ids:
                    seen_ids.add(id(val))
                    try:
                        if not _figure_has_data(val):
                            log.info("sandbox: dropping empty Plotly figure (no data in any trace)")
                            continue
                        val.update_layout(**_PREPPILOT_LAYOUT)
                        captured_plotly.append(val.to_json())
                    except Exception:
                        pass

        result_df = _find_result_df(ns, input_ids, len(df.columns))
        chart_b64 = captured_charts[0] if captured_charts else None
        chart_json = captured_plotly[0] if captured_plotly else None
        sandbox_df = ns.get("df") if isinstance(ns.get("df"), pd.DataFrame) else None

        return stdout, result_df, chart_b64, sandbox_df, chart_json

    except Exception as exc:
        log.error("sandbox execution error: %s", exc)
        return f"Code execution error: {exc}", None, None, None, None
    finally:
        if _MPL and original_show is not None:
            _plt.show = original_show  # type: ignore[method-assign]
            _plt.close("all")


def _find_result_df(
    ns: dict[str, Any],
    input_ids: set[int],
    n_input_cols: int,
) -> pd.DataFrame | None:
    """
    Scan the sandbox namespace for a new, non-empty DataFrame.
    Checks priority names first, then falls back to any other DataFrame.
    Rejects DataFrames that look like transposed Series (2 cols × n_input_cols rows).
    """
    def _valid(df: pd.DataFrame) -> bool:
        if id(df) in input_ids or len(df) == 0:
            return False
        # Reject transposed-series shape
        if len(df.columns) == 2 and len(df) == n_input_cols:
            return False
        return True

    for name in _RESULT_PRIORITY:
        val = ns.get(name)
        if isinstance(val, pd.DataFrame) and _valid(val):
            return val

    for val in ns.values():
        if isinstance(val, pd.DataFrame) and _valid(val):
            return val

    return None

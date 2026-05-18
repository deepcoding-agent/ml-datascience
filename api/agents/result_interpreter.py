"""Result Interpreter — generates human-readable responses from execution results.

Uses LLM to interpret actual computed data. NEVER produces placeholders.
Only called for codegen results — handler-only results use their summaries directly.

Self-verification (anti-hallucination):
  * `_build_ground_truth` snapshots every numeric value the codegen step
    actually produced (per-column describe() + stdout numbers).
  * The interpreter prompt embeds those numbers as a VERIFIED NUMBERS
    block and instructs the LLM to never invent numbers outside it.
  * After the LLM responds, `_verify_numbers` scans for any number in
    the reply that isn't traceable to the ground-truth set (within 1%
    tolerance). When drift is detected the function returns a list of
    suspicious tokens — the caller can choose to log/append a warning.
"""
from __future__ import annotations

import re

import pandas as pd

from api.llm import format_history_for_prompt
from api.logger import get_logger

log = get_logger(__name__)


def _build_ground_truth(final_df: pd.DataFrame | None, stdout: str) -> tuple[str, set[float]]:
    """Return (verified_block, allowed_numbers).

    verified_block is a prompt-ready snippet listing every key statistic
    available from the executed code. allowed_numbers is the set of float
    values the interpreter is permitted to quote verbatim (or derive
    obvious aggregates from — sums, ratios, counts).
    """
    lines: list[str] = []
    allowed: set[float] = set()

    def _record(value: object) -> None:
        try:
            f = float(value)
            if pd.notna(f):
                allowed.add(round(f, 6))
        except (TypeError, ValueError):
            return

    if final_df is not None and not final_df.empty:
        # Per-column statistics for numeric cols, plus value counts for low-
        # cardinality non-numeric cols. Keep it bounded — top 12 columns max.
        for col in list(final_df.columns)[:12]:
            try:
                series = final_df[col]
                if pd.api.types.is_numeric_dtype(series):
                    desc = series.describe()
                    parts = []
                    for stat in ("count", "mean", "std", "min", "25%", "50%", "75%", "max"):
                        v = desc.get(stat)
                        if v is None or pd.isna(v):
                            continue
                        _record(v)
                        parts.append(f"{stat}={float(v):.4g}")
                    if parts:
                        lines.append(f"  {col}: " + ", ".join(parts))
                else:
                    vc = series.value_counts(dropna=False).head(5)
                    parts = []
                    for k, v in vc.items():
                        _record(v)
                        parts.append(f"{k!r}={int(v)}")
                    if parts:
                        lines.append(f"  {col}: " + ", ".join(parts))
            except Exception:  # never let stat extraction break the response
                continue
        # Also record total cell count and shape — common questions reference these.
        _record(final_df.shape[0])
        _record(final_df.shape[1])

    # Pull explicit numbers out of stdout (they were genuinely printed by
    # the codegen — those are ground truth too).
    if stdout:
        for token in re.findall(r"-?\d+(?:\.\d+)?", stdout):
            _record(token)

    block = "\n".join(lines) if lines else "  (no per-column stats — short table or stdout-only)"
    return block, allowed


# Numeric tokens worth checking. Skips:
#   - tiny ordinals (0, 1, 2, …, 12)  → "first 5 rows", "12 months", section numbers
#   - thousands-separator artifacts ("1,000")
#   - non-decimal-followed numbers in code-like strings
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])")
_IGNORE_THRESHOLD = 12.0  # ignore small whole numbers — too many false positives


def _verify_numbers(reply: str, allowed: set[float], tolerance: float = 0.01) -> list[str]:
    """Return numbers in *reply* that aren't traceable to *allowed* within
    *tolerance* (1% by default). Skips tiny integers to avoid noise."""
    if not reply or not allowed:
        return []
    suspicious: list[str] = []
    for tok in _NUMBER_RE.findall(reply):
        try:
            v = float(tok)
        except ValueError:
            continue
        if abs(v) <= _IGNORE_THRESHOLD and v == int(v):
            continue
        # Direct hit (within tolerance of any allowed number, absolute OR relative).
        ok = any(
            abs(v - g) <= max(tolerance * max(abs(v), abs(g)), 1e-9)
            for g in allowed
        )
        if not ok:
            suspicious.append(tok)
    return suspicious

INTERPRETER_PROMPT = """\
The user asked: "{user_message}"

Steps executed: {steps_summary}

## CONVERSATION HISTORY (recent turns)
{conversation_history}

> History rule: use the history only when the user's current message clearly
> refers to it (follow-ups like "do the same for X", "show me more",
> "compare with the previous one"). If the question is self-contained, write
> your answer without alluding to past turns — do not mention "as before",
> "as we discussed", or summarise previous outputs unprompted.

## ACTUAL RESULTS
{actual_data}

## VERIFIED NUMBERS (ground truth — every number you quote MUST come from here)
{verified_numbers}

> Anti-hallucination rule: every numeric value (count, mean, percentage,
> ratio, etc.) in your reply MUST trace back to ACTUAL RESULTS or
> VERIFIED NUMBERS above. NEVER invent or estimate a number that isn't
> there. If a key statistic isn't listed, say so explicitly — don't make
> one up. Derived numbers (e.g. "A is 3.2x B") are fine when both A and B
> appear above.

## YOUR TASK
Write a clear, insightful response based ONLY on the actual results above.

## CONTENT RULES
1. **Lead with the answer** — directly answer the user's question first.
2. **Include key numbers** from ACTUAL RESULTS — use exact values with formatting (commas, %).
3. **Provide insight** — don't just describe what was done, explain what the results MEAN.
   - Compare values: "A is 3.8x higher than B"
   - Highlight surprises: "Interestingly, despite having fewer bedrooms..."
   - Point out patterns: "As price increases, area also increases (positive correlation)"
4. NEVER write placeholders like X%, Y%, [value], TBD, or N/A.
5. If a new dataset was created, mention row × column count.
6. **Respond in the SAME LANGUAGE as the user's question** (Thai → Thai, English → English).
7. If the user asked in Thai, respond entirely in Thai with natural, fluent language.
8. Every number MUST come from actual results. If you don't see it, don't mention it.
9. For comparisons: state the difference in both absolute and relative terms.
10. End with a brief actionable observation when relevant (e.g., "This column may benefit from normalization").

## FORMAT RULES (important — the UI renders Markdown)
- Use `## Section headings` to group related findings — e.g. `## ภาพรวม`,
  `## จุดที่น่าสนใจ`, `## ข้อแนะนำ`. Aim for 2–4 headed sections, not a single block.
- Use bullet lists (`- item`) for any enumeration of 3+ things (columns,
  segments, findings). One short sentence per bullet.
- Separate paragraphs with a blank line. NEVER produce a single wall of text.
- Use `inline code` (backticks) for column names, file names, and exact values
  the user must recognize (e.g. `COPA`, `interview_score`, `0.97`).
- Use **bold** sparingly — only key numbers or the single most important phrase
  in a section. Avoid bolding entire sentences.
- If results contain a table, render it as a Markdown table when ≤6 columns,
  otherwise highlight the most interesting rows in bullets.
- Keep paragraphs short (1–3 sentences). Scannable beats exhaustive.
"""


def interpret_final_result(
    user_message: str,
    plan: dict,
    exec_result: dict,
    llm,
    history: list | None = None,
) -> str:
    """Generate human-readable interpretation of execution results."""
    actual_data_parts: list[str] = []

    if exec_result.get("stdout"):
        actual_data_parts.append(f"STDOUT:\n{exec_result['stdout']}")

    final_df = exec_result.get("final_df")
    if final_df is not None:
        actual_data_parts.append(
            f"RESULT DATAFRAME:\n"
            f"Shape: {final_df.shape[0]} rows × {final_df.shape[1]} columns\n"
            f"First 10 rows:\n{final_df.head(10).to_string()}\n"
            f"Null counts:\n{final_df.isnull().sum().to_string()}"
        )

    if not actual_data_parts:
        actual_data_parts.append("No data output produced.")

    steps_summary = " → ".join(
        s.get("description", "") for s in plan.get("steps", [])
    )

    verified_block, allowed_numbers = _build_ground_truth(
        final_df=final_df,
        stdout=(exec_result.get("stdout") or ""),
    )

    prompt = INTERPRETER_PROMPT.format(
        user_message=user_message,
        steps_summary=steps_summary,
        actual_data="\n\n".join(actual_data_parts),
        verified_numbers=verified_block,
        conversation_history=format_history_for_prompt(history),
    )

    try:
        response = llm.invoke(prompt)
        reply = response.content.strip()
    except Exception as e:
        log.error("interpretation error: %s", e)
        return exec_result.get("stdout") or "Operation completed."

    # Soft self-verification — log suspicious numbers but don't change the
    # user-facing reply. Lets us see hallucination rate in monitoring without
    # destabilising answers when the verifier itself is wrong.
    suspicious = _verify_numbers(reply, allowed_numbers, tolerance=0.01)
    if suspicious:
        log.warning(
            "Interpreter response contains %d unverified numeric tokens: %s",
            len(suspicious),
            suspicious[:8],
        )
    return reply

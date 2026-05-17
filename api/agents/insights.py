"""Proactive insight surfacing — append 1-2 observations to the response.

After the main answer is built, a small LLM pass looks at the dataset context
plus the produced answer and surfaces things a senior data scientist would
naturally point out — data quality concerns, statistical caveats, suggested
next steps. The goal is to make the agent feel like a thoughtful partner
rather than a search box that answers literally.

Design choices:
  * The output is appended to `final_text` as a short markdown block. We
    never replace or override the agent's primary answer.
  * Insights must be SPECIFIC (cite actual numbers/columns) and ACTIONABLE
    where possible. Generic platitudes are discouraged in the prompt.
  * Strict cap: at most 2 short bullets. Long advisories drown the answer.
  * Many turns return None (no insight). Silence is preferred over filler.
"""

from __future__ import annotations

import re

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from api.logger import get_logger

_THAI_CHAR_RE = re.compile(r"[฀-๿]")

log = get_logger(__name__)


INSIGHTS_SYSTEM = """\
You are a senior data scientist reviewing what a junior analyst just produced.
The analyst answered the user's question. Your job is to ADD short, valuable
observations the user would benefit from but the literal answer might miss.

You will be given:
  * The user's question
  * Loaded datasets (variable names + column shapes)
  * The answer text the user is about to see
  * A preview of the result table (if any)

Output rules:
  * **Match the user's language.** If the user's question is in Thai, write
    the bullets in Thai. If English, write in English. Mirror their tone.
  * Return AT MOST 2 short bullets, each one sentence (~20 words max).
  * Each bullet must be SPECIFIC: cite a real column, number, or pattern
    from the data. Generic advice is forbidden.
  * Examples of GOOD insights:
      - "`area` has 12% outliers above 1.5×IQR — any mean-based comparison
        will be skewed by them."
      - "Three columns have ≥40% nulls; mean imputation would distort the
        underlying distribution significantly."
      - "Sample size (n=545) is borderline for the 13-feature regression
        you might run next — consider feature selection first."
  * Examples of BAD insights (do NOT produce these):
      - "This is interesting data." (vague)
      - "You should explore further." (filler)
      - "Make sure your data is clean." (generic)
      - "Consider visualizing." (already implied)
  * Return an EMPTY response if you have nothing concrete and useful to add.
    Silence is better than filler. Many turns should yield no insight.
  * Do NOT restate or summarise the main answer.
  * Do NOT add chart recommendations — that's the planner's job.
  * Do NOT propose further steps the user didn't ask for — the replanner
    already handled that.

Output format: markdown bullets (each starts with `- `). Nothing else.
If you have nothing to add, output an empty string.
"""


def _final_df_preview(df: pd.DataFrame | None) -> str:
    if df is None:
        return "(no result table)"
    return f"shape={df.shape}, columns={list(df.columns)}\n{df.head(3).to_string(index=False)}"


def surface_insights(
    user_message: str,
    exec_result: dict,
    df_context: str,
    final_text: str,
    llm,
) -> str | None:
    """Return a markdown block of 1-2 bullet insights, or None if nothing to add.

    The caller appends the returned text to the response when non-None.
    """
    # Skip transformations — the user is getting a new dataset, not an analysis,
    # so insights would distract from the main outcome.
    if exec_result.get("output_type") == "generate":
        return None

    # Skip very short user messages (greetings, "thanks", etc.).
    if len(user_message.strip()) < 8:
        return None

    final_df = exec_result.get("final_df")
    preview = _final_df_preview(final_df)
    answer_snippet = (final_text or "").strip()[:1200]

    user_block = (
        f"## User question\n{user_message}\n\n"
        f"## Loaded data context\n{df_context[:1500]}\n\n"
        f"## Answer the user is about to see\n{answer_snippet}\n\n"
        f"## Result table preview\n{preview}\n"
    )

    try:
        reply = llm.invoke([
            SystemMessage(content=INSIGHTS_SYSTEM),
            HumanMessage(content=user_block),
        ]).content
    except Exception as e:
        log.info("insights LLM error: %s — no insight appended", e)
        return None

    text = (reply or "").strip()
    if not text:
        return None

    # Defensive parsing: the model occasionally wraps output in fences or
    # adds a preamble. Keep only lines that look like bullet points.
    bullet_lines = [
        line for line in text.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]
    if not bullet_lines:
        return None
    bullets = "\n".join(bullet_lines[:2])  # hard cap at 2

    header = "💡 ข้อสังเกต" if _THAI_CHAR_RE.search(user_message) else "💡 Notes"
    return f"\n\n**{header}**\n{bullets}"

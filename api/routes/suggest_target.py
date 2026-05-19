"""POST /suggest-target — ask the LLM to pick the best ML target column."""
from __future__ import annotations

import json

from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from api.llm import get_llm
from api.logger import get_logger
from api.models import SuggestTargetRequest, SuggestTargetResponse

router = APIRouter()
log = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior data scientist. Identify the single best target column (dependent variable)
for a supervised machine learning task from the columns below.

Selection rules (in priority order):
1. Column names that clearly indicate the outcome to predict:
   target, label, class, y, output, result, outcome, prediction, response,
   price, cost, salary, revenue, sales, churn, survived, diagnosis, fraud,
   rating, score, grade, risk, default, approved, status, category, species.
2. If multiple candidates, prefer low-to-medium cardinality (classification)
   or continuous numeric (regression).
3. Avoid ID columns, timestamps, free-text, and obvious input features
   (age, height, weight when a health outcome is also present).
4. If nothing stands out, pick the last column (common ML convention).

Reply ONLY with valid JSON, no markdown:
{{"target_column": "<exact column name>", "reason": "<one concise sentence>"}}"""


@router.post("/suggest-target", response_model=SuggestTargetResponse)
def suggest_target(req: SuggestTargetRequest) -> SuggestTargetResponse:
    cols = req.columns
    if not cols:
        log.warning("/suggest-target called with no columns")
        return SuggestTargetResponse(target_column="", reason="No columns provided.")

    log.info("▶ /suggest-target  columns(%d)=%s", len(cols), cols[:10])

    sample_str = ""
    if req.sample_data:
        try:
            sample_str = (
                f"\n\nSample rows (first 3):\n"
                f"{json.dumps(req.sample_data[:3], default=str, indent=2)}"
            )
        except Exception:
            pass

    col_list = "\n".join(f"  - {c}" for c in cols)
    prompt = f"{_SYSTEM_PROMPT}\n\nColumns:\n{col_list}{sample_str}"

    try:
        llm  = get_llm(temperature=0.0)
        log.debug("  calling LLM for target suggestion …")
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw  = resp.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data   = json.loads(raw)
        target = str(data.get("target_column", cols[-1]))
        matched = (
            next((c for c in cols if c == target), None)
            or next((c for c in cols if c.lower() == target.lower()), None)
            or cols[-1]
        )
        log.info("✔ /suggest-target  → '%s'  reason: %s", matched, data.get("reason", ""))
        return SuggestTargetResponse(
            target_column=matched,
            reason=str(data.get("reason", "")),
        )

    except Exception as exc:
        log.error("✘ /suggest-target error: %s — falling back to last column", exc)
        return SuggestTargetResponse(
            target_column=cols[-1],
            reason="Using last column as default.",
        )

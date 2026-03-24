"""Coding agent — handles general coding / data-science Q&A (no dataset attached)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from api.llm import build_lc_history, get_llm
from api.models import ChatMessage

CODING_SYSTEM = """\
You are an expert coding and data science assistant.
Be direct — answer the question first, then explain briefly if needed.
Prefer Python. Use markdown fenced code blocks for any code.
Do NOT write long introductions. Do NOT restate the question.
Keep explanations to 2-3 sentences unless the topic genuinely requires more.
"""


def run_coding_agent(message: str, history: list[ChatMessage], model_id: str | None = None) -> str:
    llm = get_llm(temperature=0.3, max_tokens=1024, model_id=model_id)
    msgs = (
        [SystemMessage(content=CODING_SYSTEM)]
        + build_lc_history(history[-20:])
        + [HumanMessage(content=message)]
    )
    return llm.invoke(msgs).content

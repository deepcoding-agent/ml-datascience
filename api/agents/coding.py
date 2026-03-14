"""Coding agent — handles general coding / data-science Q&A (no dataset attached)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from api.llm import build_lc_history, get_llm
from api.models import ChatMessage

CODING_SYSTEM = """\
You are an expert coding and data science assistant.
Give answers that are complete but tight — cover everything that matters, skip what doesn't.
Prefer Python. Use markdown fenced code blocks for any code snippets.
End with a short 1–2 sentence summary of the key takeaway.
"""


def run_coding_agent(message: str, history: list[ChatMessage]) -> str:
    llm = get_llm(temperature=0.3)
    msgs = (
        [SystemMessage(content=CODING_SYSTEM)]
        + build_lc_history(history)
        + [HumanMessage(content=message)]
    )
    return llm.invoke(msgs).content

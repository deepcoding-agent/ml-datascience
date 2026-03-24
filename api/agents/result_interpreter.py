"""Result Interpreter — generates human-readable responses from execution results.

Uses LLM to interpret actual computed data. NEVER produces placeholders.
Only called for codegen results — handler-only results use their summaries directly.
"""
from __future__ import annotations

from api.logger import get_logger

log = get_logger(__name__)

INTERPRETER_PROMPT = """\
The user asked: "{user_message}"

Steps executed: {steps_summary}

## ACTUAL RESULTS
{actual_data}

## YOUR TASK
Write a clear, helpful response based ONLY on the actual results above.

Rules:
1. State what was done in 1-2 sentences.
2. Include key numbers/facts from ACTUAL RESULTS — use exact values.
3. NEVER write placeholders like X%, Y%, [value], TBD, or N/A.
4. If a new dataset was created, mention row count and column count.
5. If nulls were injected/filled, state the actual null count/percentage.
6. Keep response under 5 sentences unless the result is complex.
7. Respond in the SAME LANGUAGE as the user's question.
8. If results contain a table, summarize the key findings — don't repeat the whole table.
9. Every number you write MUST come from the actual results. If you don't see it, don't mention it.
"""


def interpret_final_result(
    user_message: str,
    plan: dict,
    exec_result: dict,
    llm,
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

    prompt = INTERPRETER_PROMPT.format(
        user_message=user_message,
        steps_summary=steps_summary,
        actual_data="\n\n".join(actual_data_parts),
    )

    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        log.error("interpretation error: %s", e)
        return exec_result.get("stdout") or "Operation completed."

"""Result Interpreter — generates human-readable responses from execution results.

Uses LLM to interpret actual computed data. NEVER produces placeholders.
"""
from __future__ import annotations

from api.logger import get_logger

log = get_logger(__name__)

INTERPRETER_PROMPT = """\
The user asked: "{user_message}"

The system executed these steps: {steps_summary}

ACTUAL RESULTS FROM EXECUTION:
{actual_data}

Write a clear, concise response that:
1. States what was done
2. Shows key numbers from the ACTUAL RESULTS above — use exact values
3. NEVER write X%, Y%, [value], TBD, or any placeholder
4. If a new dataset was created — mention row count and column count
5. If nulls were injected — state the actual null percentage computed
6. Keep it under 5 sentences unless result is complex
7. Respond in the same language as the user's question

IMPORTANT: Every number you write must come from ACTUAL RESULTS above.
If you don't see a number in the results — don't mention it.
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
            f"Shape: {final_df.shape[0]} rows x {final_df.shape[1]} columns\n"
            f"First 10 rows:\n{final_df.head(10).to_string()}\n"
            f"Null counts:\n{final_df.isnull().sum().to_string()}"
        )

    if not actual_data_parts:
        actual_data_parts.append("No data output produced")

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
        # Fallback: return raw stdout
        return exec_result.get("stdout") or "Operation completed."

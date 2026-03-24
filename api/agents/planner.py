"""AI Planner — chain-of-thought step planning for user requests.

The planner reads the user message and dataset context, then produces
a structured JSON plan of execution steps.  No hard-coded keyword routing.
"""
from __future__ import annotations

import json

from api.logger import get_logger

log = get_logger(__name__)

PLANNER_PROMPT = """\
You are a data science agent planner. A user has asked you to do something
with their dataset. Your job is to break this into clear executable steps.

## USER REQUEST
{user_message}

## DATASET CONTEXT
{df_context}

## AVAILABLE PRE-BUILT HANDLERS
These handlers are already implemented and can be called directly:
{available_handlers}

## YOUR TASK
Think carefully about what the user wants. Then output a JSON plan.

Rules:
1. Read the user message literally — if they say "generate", you GENERATE
2. If they say "create null", you CREATE null — do not check if nulls exist
3. If they say "make", "create", "generate", "build" → output_type = "generate"
4. If they ask a question or say "show", "plot", "visualize" → output_type = "query"
5. Always consider: should this step include a visualization?
   - If result is a distribution, count, or comparison → YES add chart step
   - If result is a single number or text answer → NO chart needed
   - If result is a new dataset → YES add preview chart step
6. Use pre-built handlers when they fit exactly
7. Generate custom code for anything not covered by handlers
8. Never refuse a task — always find a way to do it
9. If the user speaks Thai, still plan in English — Thai keywords are translated

## OUTPUT FORMAT — JSON ONLY
{{
  "understanding": "what the user wants in one sentence",
  "output_type": "query | generate",
  "steps": [
    {{
      "step_num": 1,
      "description": "what this step does",
      "use_handler": "handler_name or null",
      "handler_category": "stats | clean | transform | viz | feature | null",
      "handler_params": {{}},
      "needs_custom_code": true/false,
      "custom_code_description": "what code to write if needs_custom_code is true",
      "produces": "dataframe | chart | text | number",
      "add_visualization": true/false,
      "visualization_type": "bar | histogram | pie | scatter | line | box | heatmap | null"
    }}
  ],
  "final_output": "what the user will see at the end"
}}

## EXAMPLES

Request: "generate random null to dataset"
{{
  "understanding": "User wants to inject random null values into the dataset",
  "output_type": "generate",
  "steps": [
    {{
      "step_num": 1,
      "description": "Inject ~15% random NaN values into all columns",
      "use_handler": "inject_null",
      "handler_category": "transform",
      "handler_params": {{"value": 15}},
      "needs_custom_code": false,
      "custom_code_description": null,
      "produces": "dataframe",
      "add_visualization": true,
      "visualization_type": "bar"
    }}
  ],
  "final_output": "New dataset with random null values injected, plus a bar chart showing null counts per column"
}}

Request: "how many rows and columns"
{{
  "understanding": "User wants the shape of the dataset",
  "output_type": "query",
  "steps": [
    {{
      "step_num": 1,
      "description": "Get dataset shape (rows and columns)",
      "use_handler": "shape",
      "handler_category": "stats",
      "handler_params": {{}},
      "needs_custom_code": false,
      "custom_code_description": null,
      "produces": "text",
      "add_visualization": false,
      "visualization_type": null
    }}
  ],
  "final_output": "The dataset has N rows and M columns"
}}

Request: "split price into 5 levels and show percent of each"
{{
  "understanding": "Bin SalePrice into 5 ranges, show count and percentage per bin",
  "output_type": "query",
  "steps": [
    {{
      "step_num": 1,
      "description": "Use pd.cut to bin SalePrice into 5 equal ranges and calculate count + percentage per level",
      "use_handler": null,
      "handler_category": null,
      "handler_params": {{}},
      "needs_custom_code": true,
      "custom_code_description": "Bin SalePrice into 5 ranges with pd.cut, count each bin, calculate percentage, store as result_df with columns Range/Count/Percentage",
      "produces": "dataframe",
      "add_visualization": true,
      "visualization_type": "bar"
    }}
  ],
  "final_output": "Table and bar chart showing the percentage distribution across 5 price levels"
}}

Request: "remove duplicates then fill missing values with median"
{{
  "understanding": "Clean dataset by removing duplicate rows then filling nulls with median",
  "output_type": "generate",
  "steps": [
    {{
      "step_num": 1,
      "description": "Remove duplicate rows",
      "use_handler": "remove_duplicates",
      "handler_category": "clean",
      "handler_params": {{}},
      "needs_custom_code": false,
      "custom_code_description": null,
      "produces": "dataframe",
      "add_visualization": false,
      "visualization_type": null
    }},
    {{
      "step_num": 2,
      "description": "Fill missing values with median for numeric columns",
      "use_handler": "fill_nulls",
      "handler_category": "clean",
      "handler_params": {{"strategy": "median"}},
      "needs_custom_code": false,
      "custom_code_description": null,
      "produces": "dataframe",
      "add_visualization": false,
      "visualization_type": null
    }}
  ],
  "final_output": "Cleaned dataset with duplicates removed and missing values filled with median"
}}

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation outside JSON.
"""


def plan_steps(
    user_message: str,
    df_context: str,
    available_handlers: list[str],
    llm,
) -> dict:
    """Ask LLM to plan execution steps. Returns structured plan dict."""
    prompt = PLANNER_PROMPT.format(
        user_message=user_message,
        df_context=df_context,
        available_handlers="\n".join(f"  - {h}" for h in available_handlers),
    )

    response = llm.invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fencing if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        plan = json.loads(raw)
        log.info(
            "Plan: %s — %d step(s), output_type=%s",
            plan.get("understanding", "?")[:80],
            len(plan.get("steps", [])),
            plan.get("output_type", "?"),
        )
        return plan
    except json.JSONDecodeError:
        log.error("Planner returned invalid JSON: %s", raw[:200])
        # Fallback plan — execute as custom code
        return {
            "understanding": user_message,
            "output_type": "query",
            "steps": [
                {
                    "step_num": 1,
                    "description": f"Execute: {user_message}",
                    "use_handler": None,
                    "handler_category": None,
                    "handler_params": {},
                    "needs_custom_code": True,
                    "custom_code_description": user_message,
                    "produces": "dataframe",
                    "add_visualization": False,
                    "visualization_type": None,
                }
            ],
            "final_output": "Result of requested operation",
        }

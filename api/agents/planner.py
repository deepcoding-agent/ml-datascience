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
6. Never refuse a task — always find a way to do it
7. If the user speaks Thai, still plan in English — Thai keywords are translated

## STEP DESCRIPTION RULES — IMPORTANT

Write custom_code_description using these exact keywords when applicable
so the system can route to fast pre-built functions (instant, no LLM call):

Stats:    "get shape", "describe statistics", "null report", "data types",
          "value counts", "correlation", "skewness", "outlier report",
          "duplicate report", "unique values"

Clean:    "fill null", "fill missing", "remove duplicates",
          "fix dtypes", "drop column", "rename column"

Transform: "filter rows", "sort by", "group by", "assign value",
           "label encode", "one hot encode", "standard scale",
           "minmax normalize", "sample rows", "head", "tail"

Viz:      "bar chart", "histogram", "pie chart", "scatter plot",
          "line chart", "box plot", "heatmap", "pairplot", "violin"

For custom operations NOT in the list above (binning, moving average,
z-score, null injection, percent calculations, etc.) — describe in
plain English, code will be generated fresh.

## STEP OUTPUT TYPE

produces="chart" → set add_visualization=true, set visualization_type
produces="dataframe" → describe what the code should do
produces="text" → describe what to print/compute

## OUTPUT FORMAT — JSON ONLY
{{
  "understanding": "what the user wants in one sentence",
  "output_type": "query | generate",
  "steps": [
    {{
      "step_num": 1,
      "description": "what this step does",
      "custom_code_description": "what the Python code should do (required for dataframe/text steps)",
      "produces": "dataframe | chart | text | number",
      "add_visualization": true/false,
      "visualization_type": "bar | histogram | pie | scatter | line | box | heatmap | violin | null",
      "handler_params": {{}}
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
      "custom_code_description": "Copy df, randomly set 15% of cells in each column to NaN using np.random.choice, assign to result, print null counts and percentage",
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
      "description": "Get shape of the dataset",
      "custom_code_description": "get shape — how many rows and columns",
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
      "description": "Bin SalePrice into 5 equal ranges and calculate count + percentage per level",
      "custom_code_description": "Use pd.cut on SalePrice with bins=5, count each bin, calculate percentage, store as result with columns Range/Count/Percentage. Format bin labels as human-readable (e.g. 34K-154K)",
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
      "custom_code_description": "remove duplicate rows from dataset",
      "produces": "dataframe",
      "add_visualization": false,
      "visualization_type": null
    }},
    {{
      "step_num": 2,
      "description": "Fill missing values with median",
      "custom_code_description": "fill null values with median for numeric columns",
      "produces": "dataframe",
      "add_visualization": false,
      "visualization_type": null
    }}
  ],
  "final_output": "Cleaned dataset with duplicates removed and missing values filled with median"
}}

Request: "show pie chart of bedroom counts"
{{
  "understanding": "User wants a pie chart showing the distribution of bedroom counts",
  "output_type": "query",
  "steps": [
    {{
      "step_num": 1,
      "description": "Create pie chart of bedroom distribution",
      "produces": "chart",
      "add_visualization": true,
      "visualization_type": "pie",
      "handler_params": {{"column": "BedroomAbvGr"}}
    }}
  ],
  "final_output": "Pie chart showing bedroom count distribution"
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

"""AI Planner — produces structured JSON execution plans.

The planner reads the user message and dataset context, then outputs
a structured plan where each step explicitly specifies either a
handler (instant, 0 LLM calls) or codegen (LLM-generated code).
"""
from __future__ import annotations

import json

from api.logger import get_logger

log = get_logger(__name__)

# ── Handler catalog — embedded in the planner prompt ─────────────────────────
# This tells the LLM exactly what handlers exist, their IDs, and params.

HANDLER_CATALOG = """\
### Stats (instant query — no dataset changes)
| id | what it does | params |
|----|-------------|--------|
| stats.shape | row/column count, memory, duplicates | (none) |
| stats.describe | full profile: dtype, nulls, mean/std/min/quartiles/max for numeric, top/freq for categorical | column? |
| stats.null_report | null counts + percentages per column | (none) |
| stats.dtypes | column data types, null%, unique counts | (none) |
| stats.value_counts | top value frequencies for one column | column?, n? |
| stats.unique_values | unique value count per column | (none) |
| stats.correlation | correlation matrix + heatmap chart | (none) |
| stats.skewness | skewness per numeric column | (none) |
| stats.outlier_report | IQR-based outlier detection | (none) |
| stats.duplicate_report | duplicate row count + sample | (none) |
| stats.cross_tab | contingency table between 2 columns + heatmap | columns (list of 2), normalize? |
| stats.percentile | custom percentile report (p1, p5, p10...p99) | column?, quantiles? |
| stats.normality_test | Shapiro-Wilk normality test per column | column? |
| stats.class_balance | target class distribution + balance check | column? |
| stats.top_correlations | top N most correlated feature pairs | n? (default 10) |
| stats.kurtosis | kurtosis per numeric column | (none) |
| stats.zero_report | count zeros/empty values per column | (none) |
| stats.cardinality_report | unique ratio analysis (ID-like, binary, high, low) | (none) |

### Clean (modifies dataset → output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| clean.fill_nulls | fill missing values | column?, strategy? (auto/median/mean/mode/zero) |
| clean.remove_duplicates | remove duplicate rows | (none) |
| clean.fix_dtypes | auto-convert string→numeric/datetime | (none) |
| clean.drop_column | drop column(s) | column, columns? |
| clean.rename_column | rename a column | column (old), new_name |
| clean.strip_whitespace | trim whitespace from strings | (none) |
| clean.drop_nulls | drop rows/cols with nulls | column?, threshold? |
| clean.replace_values | replace specific values | column?, old_value, new_value |
| clean.lowercase_columns | normalize column names to snake_case | (none) |
| clean.drop_constant | drop columns with all same values | (none) |
| clean.clip_outliers | clip outliers (IQR or z-score) | column?, method? (iqr/zscore) |
| clean.remove_outliers | remove rows with outlier values | column?, method? (iqr/zscore) |
| clean.change_dtype | cast column to specific type | column, dtype (int/float/str/bool/datetime/category) |
| clean.fill_interpolate | fill nulls via interpolation | column?, method? (linear/ffill/bfill) |
| clean.lowercase_values | lowercase all string values | column? |
| clean.map_values | recode/map values in a column | column, mapping (dict e.g. {"M":"Male"}) |
| clean.reset_index | reset index to 0-based | (none) |
| clean.fill_with_value | fill nulls with specific constant | column?, value (e.g. -1, "Unknown") |
| clean.deduplicate_by | remove duplicates by specific column(s) | column?, columns?, keep? (first/last) |
| clean.drop_id_columns | auto-detect and drop ID-like columns | (none) |

### Transform (modifies dataset → output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| transform.filter | filter rows by condition | column, operator (>/</>=/<=//==/!=), value |
| transform.sort | sort by column | column, ascending? (default true) |
| transform.groupby_agg | group + aggregate | column, agg (count/sum/mean/max/min) |
| transform.assign_value | set ALL values in a column to a constant | column, value |
| transform.add_column | add new column via expression | column (name), expression |
| transform.encode_label | label-encode categoricals | column? |
| transform.encode_onehot | one-hot encode | column? |
| transform.scale_standard | z-score standardization | (none) |
| transform.scale_minmax | normalize to [0,1] | (none) |
| transform.bin_column | bin numeric column into N bins | column, n? (default 5) |
| transform.inject_null | inject random NaN values | value (number = percentage, e.g. 15 = 15%) |
| transform.sample_rows | random sample of rows | n? |
| transform.head | first N rows | n? |
| transform.tail | last N rows | n? |
| transform.pivot | pivot table (reshape) | index/column, columns?, values?, agg? (mean/sum/count) |
| transform.melt | unpivot wide→long format | id_vars?, value_vars? |
| transform.scale_robust | RobustScaler (outlier-resistant) | (none) |
| transform.nlargest | top N rows by column value | column?, n? |
| transform.nsmallest | bottom N rows by column value | column?, n? |
| transform.rank | rank values in a column | column?, ascending? |
| transform.cumulative | cumulative sum/count/max/min | column?, agg? (sum/max/min/count) |
| transform.rolling | moving average/sum/std | column?, window? (default 3), agg? (mean/sum/std) |
| transform.round_values | round numeric columns | column?, decimals? (default 2) |
| transform.split_column | split column by delimiter | column, delimiter? |
| transform.concat_columns | concatenate columns into one | columns, separator?, new_name? |
| transform.qcut | quantile-based binning (equal frequency) | column, n? (default 4) |

### Viz (charts only — output_type should be "query")
| id | what it does | params |
|----|-------------|--------|
| viz.bar_chart | bar chart of value counts | column?, percentage? |
| viz.histogram | histogram of numeric column | column? |
| viz.scatter | scatter plot | columns? (list: [x, y] or [x, y, color]) |
| viz.line_chart | line chart | column? |
| viz.box_plot | box plot | column? |
| viz.violin_plot | violin plot | column? |
| viz.heatmap | correlation heatmap | (none) |
| viz.pie_chart | pie chart (auto-groups >5 cats) | column? |
| viz.pairplot | scatter matrix | (none) |
| viz.count_plot | count/frequency bar | column? |
| viz.distribution | distribution + marginal box | column? |
| viz.missing_heatmap | missing values pattern | (none) |
| viz.treemap | treemap of categorical column | column? |
| viz.bubble_chart | bubble chart (3 numeric cols) | (none) |
| viz.stacked_bar | stacked bar (group by 2 categoricals) | columns? (list: [x, color]) |
| viz.area_chart | area chart for trends | column? |
| viz.qq_plot | QQ plot for normality check | column? |
| viz.density_plot | KDE density plot | column?, group? |
| viz.strip_plot | jitter/strip showing individual points | column?, group? |

### Feature Engineering
| id | what it does | params |
|----|-------------|--------|
| feature.feature_importance | feature importance ranking | column? (target) |
| feature.pca | principal component analysis | n? (components) |
| feature.log_transform | log transform skewed numeric cols | column? |
| feature.correlation_filter | drop highly correlated features | value? (threshold, default 0.95) |
| feature.variance_filter | drop low-variance features | value? (threshold, default 0.01) |
| feature.polynomial_features | add interaction features | columns? |
| feature.datetime_features | extract year/month/day/dayofweek/hour from datetime | column? |
| feature.target_encode | mean/target encoding for categoricals | column? (target), encode_column? |
| feature.select_k_best | select top K features by statistical test | column? (target), k? (default 10) |
| feature.power_transform | Box-Cox/Yeo-Johnson normalization | column?, method? (yeo-johnson/box-cox) |
| feature.ratio_features | create ratio features (col_a/col_b) | columns? |
| feature.frequency_encode | encode categoricals by frequency | column? |
| feature.cyclical_encode | sin/cos encoding for cyclical features | column, period? |
| feature.sqrt_transform | square root transform for moderate skew | column? |
| feature.mutual_info | mutual information scores (numeric + categorical) | column? (target) |
| feature.lag_features | create lag/shift features for time series | column?, lags? (list or int, default [1,2,3]) |
| feature.text_features | extract text stats: length, word count, digit count, uppercase ratio | column? |
| feature.quantile_transform | map values to uniform/normal distribution | column?, distribution? (normal/uniform) |
| feature.diff_features | create difference features (first/second order) | column?, periods? (default 1) |
| feature.aggregation_features | group-by stats as new features (mean/std/count) | column? (group-by col), agg_column? |
"""

PLANNER_PROMPT = """\
You are the planner for a data-science agent. Given a user request and dataset,
output a JSON execution plan.

## USER REQUEST
{user_message}

## DATASET
{df_context}

## AVAILABLE HANDLERS (use whenever possible — instant, no code generation)
{handler_catalog}

## DECISION RULES — READ CAREFULLY

### When to use a handler
ALWAYS prefer a handler when one fits. Handlers are instant, reliable, and tested.
Scan the handler table above — if ANY handler matches the user's intent, use it.

### When to use codegen
Use codegen ONLY when NO handler in the table above can do it:
  - Binning/cutting with custom labels (pd.cut with formatting)
  - Custom calculations (percentages, ratios, derived metrics)
  - Pivot, melt, reshape, merge, join
  - Moving average, rolling window, cumulative operations
  - Complex multi-condition filtering
  - Any custom math/logic not covered by a handler

### output_type
  - "query" → stats, viz, questions, any read-only operation
  - "generate" → cleaning, transforms, data generation — anything that creates/modifies data

### Smart chart selection — CHOOSE THE RIGHT CHART TYPE
When user asks for a plot/chart/visualization without specifying exact type, pick the BEST chart:
- "percent" / "percentage" / "proportion" / "share" / "ratio" of a categorical column (≤10 unique) → **viz.pie_chart** (NOT bar_chart)
- "distribution" / "spread" of a numeric column → **viz.distribution** or **viz.histogram**
- "compare" / "comparison" across categories → **viz.bar_chart**
- "trend" / "over time" / time series → **viz.line_chart**
- "relationship" / "vs" / "between" two numeric columns → **viz.scatter**
- "correlation" → **viz.heatmap** or **stats.correlation**
- "outliers" / "spread" comparison across groups → **viz.box_plot**
- "counts" / "frequency" / "how many" → **viz.count_plot** or **viz.bar_chart**
- If user says "plot" or "chart" generically for a column:
  - categorical with ≤6 unique → viz.pie_chart
  - categorical with 7-20 unique → viz.bar_chart
  - numeric → viz.histogram

### Other rules
1. Column names MUST match actual columns from DATASET section — NEVER invent column names.
2. Keep plans SHORT: 1 step if possible, 2-3 for multi-part, max 5 steps.
3. Each step has EITHER "handler" OR "codegen" — never both.
4. For "tell me about the dataset" / "overview" / "info" / "describe" / "summary" → stats.describe
5. For "show nulls" / "missing values" / "how many nulls" → stats.null_report
6. For "fill nulls" / "fill missing" → clean.fill_nulls (NOT stats.null_report)
7. For "inject/create/generate nulls" → transform.inject_null
8. For any chart request → use the matching viz handler (apply smart chart selection above)
9. For "correlation" (no chart word) → stats.correlation
10. For "correlation heatmap" → viz.heatmap
11. If user speaks Thai, translate intent to English and plan normally.

## OUTPUT FORMAT — valid JSON, no markdown fences, no explanation

For handler steps:
{{"step_num":1,"description":"...","handler":{{"id":"category.sub","params":{{}}}}}}

For codegen steps:
{{"step_num":2,"description":"...","codegen":{{"task":"detailed Python task description","produces":"dataframe|chart|text"}}}}

Full format:
{{
  "understanding": "one sentence",
  "output_type": "query | generate",
  "steps": [ ... ]
}}

## EXAMPLES

User: "how many rows and columns"
{{"understanding":"Get dataset dimensions","output_type":"query","steps":[{{"step_num":1,"description":"Get dataset shape","handler":{{"id":"stats.shape","params":{{}}}}}}]}}

User: "fill missing values with median"
{{"understanding":"Fill all nulls using median","output_type":"generate","steps":[{{"step_num":1,"description":"Fill nulls with median","handler":{{"id":"clean.fill_nulls","params":{{"strategy":"median"}}}}}}]}}

User: "show bar chart of bedroom counts"
{{"understanding":"Bar chart of bedroom distribution","output_type":"query","steps":[{{"step_num":1,"description":"Bar chart of bedrooms","handler":{{"id":"viz.bar_chart","params":{{"column":"BedroomAbvGr"}}}}}}]}}

User: "correlation heatmap"
{{"understanding":"Show correlation heatmap","output_type":"query","steps":[{{"step_num":1,"description":"Correlation heatmap","handler":{{"id":"viz.heatmap","params":{{}}}}}}]}}

User: "inject 15% random nulls"
{{"understanding":"Inject random null values","output_type":"generate","steps":[{{"step_num":1,"description":"Inject 15% random nulls","handler":{{"id":"transform.inject_null","params":{{"value":15}}}}}}]}}

User: "remove duplicates then fill missing values"
{{"understanding":"Clean: deduplicate then fill nulls","output_type":"generate","steps":[{{"step_num":1,"description":"Remove duplicates","handler":{{"id":"clean.remove_duplicates","params":{{}}}}}},{{"step_num":2,"description":"Fill missing values","handler":{{"id":"clean.fill_nulls","params":{{"strategy":"auto"}}}}}}]}}

User: "split price into 5 levels and show percentage"
{{"understanding":"Bin price into 5 ranges with percentages","output_type":"query","steps":[{{"step_num":1,"description":"Bin price into 5 ranges with count and percentage","codegen":{{"task":"Use pd.cut on the price column with bins=5. Count each bin, calculate percentage. Create result DataFrame with Range/Count/Percentage. Format bin labels as human-readable (34K-154K). Create bar chart: fig = px.bar(result, x='Range', y='Count', title='Price Distribution', text='Count')","produces":"dataframe"}}}}]}}

User: "describe the dataset"
{{"understanding":"Show descriptive statistics","output_type":"query","steps":[{{"step_num":1,"description":"Descriptive statistics","handler":{{"id":"stats.describe","params":{{}}}}}}]}}

User: "set all values of MSSubClass to 111"
{{"understanding":"Assign constant value to column","output_type":"generate","steps":[{{"step_num":1,"description":"Set MSSubClass to 111","handler":{{"id":"transform.assign_value","params":{{"column":"MSSubClass","value":111}}}}}}]}}

User: "show me nulls"
{{"understanding":"Check missing values","output_type":"query","steps":[{{"step_num":1,"description":"Null report","handler":{{"id":"stats.null_report","params":{{}}}}}}]}}

User: "generate random null 15%"
{{"understanding":"Inject 15% random nulls","output_type":"generate","steps":[{{"step_num":1,"description":"Inject 15% random nulls","handler":{{"id":"transform.inject_null","params":{{"value":15}}}}}}]}}

User: "บอกข้อมูลของ data ชุดนี้" (tell me about this dataset)
{{"understanding":"Dataset overview","output_type":"query","steps":[{{"step_num":1,"description":"Descriptive statistics","handler":{{"id":"stats.describe","params":{{}}}}}}]}}

User: "เปลี่ยนค่า MSSubClass ให้เป็น 999" (change MSSubClass to 999)
{{"understanding":"Set MSSubClass to 999","output_type":"generate","steps":[{{"step_num":1,"description":"Assign 999 to MSSubClass","handler":{{"id":"transform.assign_value","params":{{"column":"MSSubClass","value":999}}}}}}]}}

User: "sort by price descending"
{{"understanding":"Sort by price descending","output_type":"generate","steps":[{{"step_num":1,"description":"Sort by SalePrice descending","handler":{{"id":"transform.sort","params":{{"column":"SalePrice","ascending":false}}}}}}]}}

User: "show top 5 neighborhoods by count"
{{"understanding":"Value counts of neighborhoods","output_type":"query","steps":[{{"step_num":1,"description":"Top 5 neighborhoods","handler":{{"id":"stats.value_counts","params":{{"column":"Neighborhood","n":5}}}}}}]}}

User: "label encode all categorical columns"
{{"understanding":"Label encode categoricals","output_type":"generate","steps":[{{"step_num":1,"description":"Label encode categorical columns","handler":{{"id":"transform.encode_label","params":{{}}}}}}]}}

User: "filter rows where price > 200000"
{{"understanding":"Filter expensive houses","output_type":"generate","steps":[{{"step_num":1,"description":"Filter SalePrice > 200000","handler":{{"id":"transform.filter","params":{{"column":"SalePrice","operator":">","value":200000}}}}}}]}}

User: "plot percent of bedrooms"
{{"understanding":"Pie chart of bedroom percentage distribution","output_type":"query","steps":[{{"step_num":1,"description":"Pie chart of bedrooms percentage","handler":{{"id":"viz.pie_chart","params":{{"column":"bedrooms"}}}}}}]}}

User: "show distribution of price"
{{"understanding":"Price distribution histogram","output_type":"query","steps":[{{"step_num":1,"description":"Distribution of price","handler":{{"id":"viz.distribution","params":{{"column":"price"}}}}}}]}}

User: "plot price vs area"
{{"understanding":"Scatter plot of price vs area","output_type":"query","steps":[{{"step_num":1,"description":"Scatter price vs area","handler":{{"id":"viz.scatter","params":{{"columns":["area","price"]}}}}}}]}}

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation, no code fences.
"""


def plan_steps(
    user_message: str,
    df_context: str,
    llm,
) -> dict:
    """Ask LLM to plan execution steps. Returns structured plan dict."""
    prompt = PLANNER_PROMPT.format(
        user_message=user_message,
        df_context=df_context,
        handler_catalog=HANDLER_CATALOG,
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
        log.error("Planner returned invalid JSON: %s", raw[:300])
        return {
            "understanding": user_message,
            "output_type": "query",
            "steps": [
                {
                    "step_num": 1,
                    "description": user_message,
                    "codegen": {
                        "task": user_message,
                        "produces": "text",
                    },
                }
            ],
        }

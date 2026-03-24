# Handler Feature Reference — 101 Pre-built Handlers

Complete reference for all pre-built handlers in the PrepPilot DS-Agent.
The AI planner selects handlers by `category.id` and passes params as JSON.

---

## Stats (18 handlers)

Instant read-only queries — no dataset modification.

| ID | Description | Params |
|----|-------------|--------|
| `stats.describe` | Full column profile: dtype, nulls, mean/std/min/quartiles/max (numeric), top/freq (categorical) | `column?` |
| `stats.shape` | Row/column count, memory, numeric/categorical breakdown, nulls, duplicates | — |
| `stats.null_report` | Null count + percentage per column, sorted by severity | — |
| `stats.dtypes` | Column data types with null% and unique counts | — |
| `stats.value_counts` | Top N value frequencies for a column | `column?`, `n?` (default 10) |
| `stats.unique_values` | Unique count per column sorted descending | — |
| `stats.correlation` | Correlation matrix + heatmap chart | — |
| `stats.top_correlations` | Top N most correlated feature pairs (easier to read than matrix) | `n?` (default 10) |
| `stats.cross_tab` | Contingency table between 2 columns + heatmap chart | `columns` (list of 2), `normalize?` |
| `stats.class_balance` | Target class distribution + bar chart + balance assessment | `column?` |
| `stats.normality_test` | Shapiro-Wilk test per numeric column with skewness + kurtosis | `column?` |
| `stats.skewness` | Skewness per numeric column sorted by magnitude | — |
| `stats.kurtosis` | Kurtosis per numeric column with shape label (heavy/light-tailed) | — |
| `stats.percentile` | Custom percentile report (p1, p5, p10, p25, p50, p75, p90, p95, p99) | `column?`, `quantiles?` |
| `stats.outlier_report` | IQR-based outlier count per column with bounds | — |
| `stats.duplicate_report` | Duplicate row count + sample of duplicates | — |
| `stats.zero_report` | Count of zero/empty values per column | — |
| `stats.cardinality_report` | Unique ratio analysis — labels each column as ID-like/Binary/High/Low/Medium | — |

---

## Clean (20 handlers)

Data cleaning operations — all produce `output_type: "generate"`.

### Null Handling

| ID | Description | Params |
|----|-------------|--------|
| `clean.fill_nulls` | Fill nulls with strategy: auto (skew-aware), median, mean, mode, or zero | `column?`, `strategy?` |
| `clean.fill_with_value` | Fill nulls with a specific constant (e.g. -1, "Unknown", "N/A") | `column?`, `value` |
| `clean.fill_interpolate` | Fill nulls via interpolation: linear, forward fill (ffill), backward fill (bfill) | `column?`, `method?` |
| `clean.drop_nulls` | Drop rows with nulls; optionally drop high-null columns first (threshold) | `column?`, `threshold?` |

### Duplicates & Constants

| ID | Description | Params |
|----|-------------|--------|
| `clean.remove_duplicates` | Remove all duplicate rows | — |
| `clean.deduplicate_by` | Remove duplicates by specific column(s), keep first or last | `column?`, `columns?`, `keep?` |
| `clean.drop_constant` | Drop columns where all values are identical (zero information) | — |

### Outliers

| ID | Description | Params |
|----|-------------|--------|
| `clean.clip_outliers` | Cap outlier values using IQR or z-score (values are clipped, not removed) | `column?`, `method?` |
| `clean.remove_outliers` | Remove rows containing outlier values (IQR or z-score) | `column?`, `method?` |

### Column Operations

| ID | Description | Params |
|----|-------------|--------|
| `clean.drop_column` | Drop one or more columns | `column`, `columns?` |
| `clean.drop_id_columns` | Auto-detect and drop ID-like columns (sequential, high unique ratio, generic names) | — |
| `clean.rename_column` | Rename a column | `column` (old), `new_name` |
| `clean.lowercase_columns` | Normalize all column names to snake_case (CamelCase → snake_case) | — |
| `clean.reset_index` | Reset DataFrame index to 0-based sequential | — |

### Type Conversion

| ID | Description | Params |
|----|-------------|--------|
| `clean.fix_dtypes` | Auto-detect and convert string columns to numeric or datetime | — |
| `clean.change_dtype` | Manually cast a column to: int, float, str, bool, datetime, category | `column`, `dtype` |

### Value Operations

| ID | Description | Params |
|----|-------------|--------|
| `clean.replace_values` | Replace specific values (e.g. "?" → NaN) in one or all columns | `column?`, `old_value`, `new_value` |
| `clean.map_values` | Recode/map values using a dict (e.g. {"M": "Male", "F": "Female"}) | `column`, `mapping` |
| `clean.lowercase_values` | Lowercase all string values in column(s) | `column?` |
| `clean.strip_whitespace` | Trim leading/trailing whitespace from all string columns | — |

---

## Transform (26 handlers)

Data transformation and reshaping — most produce `output_type: "generate"`.

### Reshaping

| ID | Description | Params |
|----|-------------|--------|
| `transform.pivot` | Pivot table (aggregate + reshape) | `index`/`column`, `columns?`, `values?`, `agg?` |
| `transform.melt` | Unpivot wide → long format | `id_vars?`, `value_vars?` |
| `transform.split_column` | Split a string column by delimiter into multiple new columns | `column`, `delimiter?` |
| `transform.concat_columns` | Concatenate multiple columns into one new column | `columns`, `separator?`, `new_name?` |

### Filtering & Sorting

| ID | Description | Params |
|----|-------------|--------|
| `transform.filter` | Filter rows by condition (==, !=, >, <, >=, <=) | `column`, `operator`, `value` |
| `transform.sort` | Sort by column ascending or descending | `column`, `ascending?` |
| `transform.nlargest` | Top N rows by column value | `column?`, `n?` |
| `transform.nsmallest` | Bottom N rows by column value | `column?`, `n?` |
| `transform.head` | First N rows | `n?` (default 10) |
| `transform.tail` | Last N rows | `n?` (default 10) |
| `transform.sample_rows` | Random sample of N rows | `n?` (default 10) |

### Aggregation & Calculation

| ID | Description | Params |
|----|-------------|--------|
| `transform.groupby_agg` | Group by column + aggregate (count/sum/mean/max/min) | `column`, `agg?` |
| `transform.rank` | Add rank column based on numeric values | `column?`, `ascending?` |
| `transform.cumulative` | Cumulative sum/count/max/min | `column?`, `agg?` |
| `transform.rolling` | Rolling/moving window: mean, sum, std | `column?`, `window?`, `agg?` |
| `transform.round_values` | Round numeric columns to N decimal places | `column?`, `decimals?` |
| `transform.add_column` | Add new column via pandas eval expression | `column` (name), `expression` |
| `transform.assign_value` | Set all values in a column to a constant | `column`, `value` |

### Encoding

| ID | Description | Params |
|----|-------------|--------|
| `transform.encode_label` | Label-encode categorical columns (A→0, B→1, C→2) | `column?` |
| `transform.encode_onehot` | One-hot encode categorical columns (drop_first=True) | `column?` |

### Scaling

| ID | Description | Params |
|----|-------------|--------|
| `transform.scale_standard` | StandardScaler: z-score normalization (mean=0, std=1) | — |
| `transform.scale_minmax` | MinMaxScaler: normalize to [0, 1] | — |
| `transform.scale_robust` | RobustScaler: median-centered, IQR-scaled (resistant to outliers) | — |

### Binning

| ID | Description | Params |
|----|-------------|--------|
| `transform.bin_column` | Equal-width binning into N bins | `column`, `n?` (default 5) |
| `transform.qcut` | Quantile-based binning (equal-frequency bins) | `column`, `n?` (default 4) |

### Other

| ID | Description | Params |
|----|-------------|--------|
| `transform.inject_null` | Inject random NaN values (for testing) | `value` (percentage, e.g. 15 = 15%) |

---

## Visualization (22 handlers)

All charts use Plotly with a unified minimal theme (plotly_white, `#FB8C3C` accent, Inter font).

### Basic Charts

| ID | Description | Params |
|----|-------------|--------|
| `viz.bar_chart` | Bar chart of value counts (top 15) | `column?`, `percentage?` |
| `viz.stacked_bar` | Stacked bar chart — group by one column, color by another | `columns?` (list: [x, color]) |
| `viz.histogram` | Histogram with marginal box plot | `column?` |
| `viz.pie_chart` | Donut pie chart (auto-groups >6 categories into "Other") | `column?` |
| `viz.count_plot` | Count/frequency bar chart (top 20) | `column?` |
| `viz.line_chart` | Line chart | `column?` |
| `viz.area_chart` | Area chart for trends | `column?` |
| `viz.scatter` | Scatter plot (optional color dimension) | `columns?` (list: [x, y, color?]) |

### Distribution & Statistical

| ID | Description | Params |
|----|-------------|--------|
| `viz.distribution` | Histogram + marginal box plot | `column?` |
| `viz.density_plot` | KDE density plot (optionally grouped) | `column?`, `group?` |
| `viz.box_plot` | Box plot (single column or multi-column comparison) | `column?` |
| `viz.violin_plot` | Violin plot with embedded box | `column?` |
| `viz.strip_plot` | Jitter/strip plot showing individual data points | `column?`, `group?` |
| `viz.qq_plot` | QQ plot for normality check (data vs normal line) | `column?` |

### Correlation & Multi-variable

| ID | Description | Params |
|----|-------------|--------|
| `viz.heatmap` | Correlation heatmap (RdYlBu color scale) | — |
| `viz.pairplot` | Scatter matrix of numeric columns (max 5) | — |
| `viz.parallel_coords` | Parallel coordinates plot (max 6 numeric cols) | — |
| `viz.bubble_chart` | Bubble chart using 3 numeric columns (x, y, size) | — |

### Categorical & Hierarchical

| ID | Description | Params |
|----|-------------|--------|
| `viz.treemap` | Treemap of categorical column value counts | `column?` |
| `viz.sunburst` | Sunburst chart for 2 categorical columns | — |

### Missing Data

| ID | Description | Params |
|----|-------------|--------|
| `viz.missing_heatmap` | Missing values pattern heatmap | — |

### Time Series

| ID | Description | Params |
|----|-------------|--------|
| `viz.time_series` | Time series line chart (requires datetime column) | `column?` |

---

## Feature Engineering (15 handlers)

Feature creation, selection, and transformation — most produce `output_type: "generate"`.

### Feature Selection

| ID | Description | Params |
|----|-------------|--------|
| `feature.feature_importance` | Random Forest feature importance ranking + horizontal bar chart | `column?` (target) |
| `feature.mutual_info` | Mutual information scores (works for numeric + categorical) + bar chart | `column?` (target) |
| `feature.select_k_best` | Select top K features using f_classif/f_regression + chart | `column?` (target), `k?` |
| `feature.correlation_filter` | Drop features with correlation above threshold | `value?` (default 0.95) |
| `feature.variance_filter` | Drop features with variance below threshold | `value?` (default 0.01) |

### Dimensionality Reduction

| ID | Description | Params |
|----|-------------|--------|
| `feature.pca` | PCA + scatter plot with explained variance | `n?` (components, default 2) |

### Transforms

| ID | Description | Params |
|----|-------------|--------|
| `feature.log_transform` | log1p transform for highly skewed columns (skew > 1) | `column?` |
| `feature.sqrt_transform` | Square root transform for moderately skewed columns (skew > 0.5) | `column?` |
| `feature.power_transform` | Yeo-Johnson or Box-Cox power transform (standardized) | `column?`, `method?` |

### Encoding

| ID | Description | Params |
|----|-------------|--------|
| `feature.target_encode` | Mean/target encoding for high-cardinality categoricals | `column?` (target), `encode_column?` |
| `feature.frequency_encode` | Encode categorical values by their frequency (normalized) | `column?` |
| `feature.cyclical_encode` | sin/cos encoding for cyclical features (month, dayofweek, hour) | `column`, `period?` |

### Feature Creation

| ID | Description | Params |
|----|-------------|--------|
| `feature.datetime_features` | Extract year, month, day, dayofweek, hour from datetime columns | `column?` |
| `feature.ratio_features` | Create ratio features between numeric column pairs (col_a / col_b) | `columns?` |
| `feature.polynomial_features` | Create interaction features (col_a × col_b) for numeric columns | `columns?` |

---

## Chart Theme

All handler charts use a shared minimal theme:

- **Template**: `plotly_white`
- **Font**: Inter, Noto Sans Thai, Tahoma
- **Accent color**: `#FB8C3C` (PrepPilot orange)
- **Color palette**: `#FB8C3C`, `#2EC4B6`, `#457B9D`, `#E71D36`, `#FF9F1C`, `#A8DADC`, `#1D3557`, `#6B4226`
- **Background**: transparent
- **Grid**: no x-axis grid, subtle `#F0F0F0` y-axis grid
- **Bar charts**: text labels outside, 0.3 gap

Codegen charts from the sandbox also receive the same theme via `_PREPPILOT_LAYOUT`.

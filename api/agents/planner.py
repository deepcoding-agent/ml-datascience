"""AI Planner — two-stage routing with 14 sub-categories.

Stage 1 (Router): Lightweight LLM call classifies message into
  1-3 sub-categories from 14 options (instead of 7).
  ~200 tokens output.

Stage 2 (Planner): Full planner only sees handlers from selected
  sub-categories (~25-35 handlers instead of 56-64). Much more accurate.
"""
from __future__ import annotations

import json
import re as _re

from api.logger import get_logger

log = get_logger(__name__)

# ── Handler catalog — split into 14 sub-categories ──────────────────────────

HANDLER_CATALOG = """\
### Stats Summary (instant query — descriptive statistics, profiles, reports)
| id | what it does | output | params |
|----|-------------|--------|--------|
| stats.shape | row/column count, memory usage, duplicate count | table | (none) |
| stats.describe | full profile: dtype, nulls, mean/std/min/quartiles/max per column | table | column? |
| stats.null_report | null counts + percentages per column, sorted by severity | table | (none) |
| stats.dtypes | column data types with null% and unique counts | table | (none) |
| stats.value_counts | value frequencies + PERCENTAGE for one column — use for "กี่เปอร์เซ็นต์/how many percent/proportion" | table | column?, n? |
| stats.unique_values | unique value count per column | table | (none) |
| stats.skewness | skewness per numeric column (detects asymmetry) | table | (none) |
| stats.outlier_report | IQR-based outlier detection with count per column | table | (none) |
| stats.duplicate_report | duplicate row count + sample of duplicated rows | table | (none) |
| stats.cross_tab | contingency table (crosstab) between 2 columns + heatmap chart | table+chart | columns (list of 2), normalize? |
| stats.percentile | custom percentile report (p1, p5, p10...p99) | table | column?, quantiles? |
| stats.class_balance | target class distribution + imbalance severity | table | column? |
| stats.top_correlations | top N most correlated pairs with strength label | table | n? (default 10) |
| stats.kurtosis | kurtosis per numeric column | table | (none) |
| stats.zero_report | count zeros/empty values per column | table | (none) |
| stats.cardinality_report | unique ratio analysis (ID-like, binary, high, low cardinality) | table | (none) |
| stats.mode_report | most frequent value per column | table | (none) |
| stats.variance_report | variance per numeric column | table | (none) |
| stats.range_report | range (max-min) per numeric column | table | (none) |
| stats.iqr_report | interquartile range per numeric column | table | (none) |
| stats.z_score_report | z-score analysis, flags extreme values (|z|>3) | table | column? |
| stats.frequency_table | frequency table with cumulative percentage — use for "สัดส่วน/breakdown" | table | column? |
| stats.coefficient_variation | coefficient of variation (CV) — compare spread across columns | table | (none) |
| stats.correlation_rank | Spearman rank correlation matrix + heatmap chart | table+chart | (none) |
| stats.entropy_report | Shannon entropy per column (measures randomness) | table | (none) |
| stats.gini_report | Gini impurity per categorical column | table | (none) |
| stats.missing_pattern | which columns tend to be missing together (co-occurrence) | table | (none) |
| stats.quantile_detail | detailed quantiles (1,5,10,25,50,75,90,95,99) | table | column? |
| stats.group_stats | descriptive stats per group — use for "แต่ละกลุ่มมีค่า X เท่าไหร่" | table | column? (group col), value_column? |
| stats.column_compare | compare two columns: mean, std, range, distribution | table | columns (list of 2) |
| stats.mutual_info_report | mutual information scores — which features are most predictive | table | column? (target) |
| stats.summary_extended | extended summary: mean,median,mode,std,var,range,IQR,skew,kurt | table | (none) |
| stats.ratio_report | compute key ratios between numeric columns | table | (none) |
| stats.top_bottom_values | top N and bottom N values — use for "สูงสุด/ต่ำสุด/max/min" | table | column?, n? (default 5) |
| stats.memory_report | detailed memory usage per column in KB/MB | table | (none) |
| stats.sparsity_report | sparsity analysis (zeros + nulls percentage per column) | table | (none) |
| stats.data_sample | random sample of N rows with basic stats | table | n? (default 10) |
| stats.class_weight_calc | calculate balanced class weights for ML training (sklearn-compatible) | table | column (target) |

### Stats Test (statistical tests, hypothesis testing, correlation)
| id | what it does | params |
|----|-------------|--------|
| stats.correlation | correlation matrix + heatmap chart | (none) |
| stats.normality_test | Shapiro-Wilk normality test per column | column? |
| stats.chi2_test | chi-squared independence test between two categorical columns | columns (list of 2) |
| stats.t_test | independent t-test: compare numeric column across two groups | column? (numeric), group_column? (categorical) |
| stats.anova_test | one-way ANOVA: compare numeric across multiple groups | column? (numeric), group_column? (categorical) |
| stats.mann_whitney | Mann-Whitney U non-parametric test | column? (numeric), group_column? (categorical) |
| stats.ks_test | Kolmogorov-Smirnov normality test | column? |
| stats.distribution_fit | fit best distribution (normal/lognormal/exponential) | column? |
| stats.stability_report | check feature stability (split data, compare halves) | (none) |
| stats.pairwise_stats | pairwise comparison between numeric columns | (none) |
| stats.time_stats | time-series stats (autocorrelation, stationarity hint) | column? |
| stats.cluster_tendency | Hopkins statistic — is data clusterable? | (none) |
| stats.normality_comprehensive | comprehensive normality: Shapiro + D'Agostino + Anderson-Darling | column? |
| stats.covariance | compute covariance matrix between numeric columns | (none) |
| stats.confidence_interval | confidence intervals for numeric columns | column?, confidence? (default 0.95) |
| stats.cramers_v | Cramer's V association between categorical columns | (none) |
| stats.point_biserial | point-biserial correlation between binary and numeric cols | (none) |
| stats.levene_test | Levene's test for equality of variances across groups | column?, group? |

### Clean Values (fill/fix/remove values — output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| clean.fill_nulls | fill missing values | column?, strategy? (auto/median/mean/mode/zero) |
| clean.drop_nulls | drop rows/cols with nulls | column?, threshold? |
| clean.fill_interpolate | fill nulls via interpolation | column?, method? (linear/ffill/bfill) |
| clean.fill_with_value | fill nulls with specific constant | column?, value (e.g. -1, "Unknown") |
| clean.fill_mode | fill nulls with mode (most frequent value) | column? |
| clean.fill_forward_backward | ffill then bfill to fill nulls | column? |
| clean.fill_median_by_group | fill nulls with group-level median | column (group col), value_column? |
| clean.fill_with_distribution | fill nulls by sampling from column distribution | column?, seed? |
| clean.replace_values | replace specific values | column?, old_value, new_value |
| clean.map_values | recode/map values in a column | column, mapping (dict e.g. {"M":"Male"}) |
| clean.clip_outliers | clip outliers (IQR or z-score) | column?, method? (iqr/zscore) |
| clean.remove_outliers | remove rows with outlier values | column?, method? (iqr/zscore) |
| clean.cap_outliers_percentile | cap at Nth percentile | column?, lower? (default 1), upper? (default 99) |
| clean.remove_empty_rows | remove rows where all values are null/empty | (none) |
| clean.remove_zero_rows | remove rows where column(s) are zero | column? |
| clean.remove_negative | remove rows with negative values | column? |
| clean.round_to_nearest | round numeric values to nearest N (5, 10, 100, etc.) | column, nearest? (default 10) |
| clean.coalesce_columns | merge columns taking first non-null value per row | columns (list), new_name? |
| clean.fix_date_outliers | remove/clip dates outside valid range | column, min_date?, max_date?, action? (remove/clip) |
| clean.standardize_dates | parse mixed date formats to consistent format | column?, format? (default %Y-%m-%d) |
| clean.standardize_categories | merge similar categories (strip, lower, map) | column, mapping? (dict) |
| clean.remove_rare_categories | replace categories with < N occurrences with "Other" | column, min_count? (default 5), replacement? |
| clean.dedup_keep_latest | dedup by column keeping latest by sort column | column (key), date_column (sort) |
| clean.clean_currency | clean currency strings ($, €, ¥, commas) → float | column? |
| clean.smote_oversample | SMOTE oversampling — synthetic minority samples via k-NN | column (target), k_neighbors? (default 5) |
| clean.random_oversample | randomly duplicate minority class rows | column (target) |
| clean.random_undersample | randomly remove majority class rows | column (target) |
| clean.adasyn_oversample | ADASYN adaptive oversampling — focuses on harder examples | column (target), n_neighbors? (default 5) |

### Clean Structure (fix types, remove columns, deduplicate, standardize — output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| clean.remove_duplicates | remove duplicate rows | (none) |
| clean.fix_dtypes | auto-convert string→numeric/datetime | (none) |
| clean.drop_column | drop column(s) | column, columns? |
| clean.rename_column | rename a column | column (old), new_name |
| clean.strip_whitespace | trim whitespace from strings | (none) |
| clean.lowercase_columns | normalize column names to snake_case | (none) |
| clean.drop_constant | drop columns with all same values | (none) |
| clean.change_dtype | cast column to specific type | column, dtype (int/float/str/bool/datetime/category) |
| clean.lowercase_values | lowercase all string values | column? |
| clean.reset_index | reset index to 0-based | (none) |
| clean.deduplicate_by | remove duplicates by specific column(s) | column?, columns?, keep? (first/last) |
| clean.drop_id_columns | auto-detect and drop ID-like columns | (none) |
| clean.fix_numeric_strings | convert "$1,234" / "1.234,56" to numeric | column? |
| clean.clean_column_names | remove special chars, spaces→underscore, lowercase | (none) |
| clean.fix_boolean | standardize yes/no/true/false/Y/N/1/0 → bool | column? |
| clean.fix_encoding | fix mojibake/encoding issues (special chars) | column? |
| clean.remove_html_tags | strip HTML/XML tags from strings | column? |
| clean.remove_non_ascii | remove non-ASCII characters | column? |
| clean.remove_special_chars | remove special characters (keep alphanumeric + spaces) | column?, keep? (regex pattern) |
| clean.normalize_text_case | normalize to title/upper/lower/sentence case | column?, case? (lower/upper/title/sentence) |
| clean.remove_high_null_cols | drop columns above null threshold | threshold? (default 0.5) |
| clean.clean_phone_numbers | standardize phone numbers to digits-only | column |
| clean.split_name | split "John Doe" → first_name, last_name | column |
| clean.fix_whitespace_names | fix " John  Doe " → "John Doe" | column? |
| clean.remove_urls | remove URLs from text | column? |
| clean.remove_emails | remove email addresses from text | column? |
| clean.fix_mixed_types | convert mixed-type columns to consistent type | column? |
| clean.clean_text_whitespace | normalize all whitespace (double spaces, tabs, newlines) | column? |
| clean.anonymize_column | hash/mask/redact a column for PII protection | column, method? (hash/mask/redact) |
| clean.parse_json_column | parse JSON strings in a column into separate columns | column |
| clean.trim_text_length | truncate text to max character length | column, max_length? (default 100) |
| clean.auto_dtype_infer | smart auto-detect and convert column dtypes (numeric, bool, datetime, category) | (none) |

### Transform Reshape (filter, sort, group, pivot, merge, reshape — output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| transform.filter | filter rows by condition | column, operator (>/</>=/<=//==/!=), value |
| transform.sort | sort by column | column, ascending? (default true) |
| transform.groupby_agg | group + aggregate | column, agg (count/sum/mean/max/min) |
| transform.add_column | add new column via expression | column (name), expression |
| transform.assign_value | set ALL values in a column to a constant | column, value |
| transform.pivot | pivot table (reshape) | index/column, columns?, values?, agg? (mean/sum/count) |
| transform.melt | unpivot wide→long format | id_vars?, value_vars? |
| transform.merge | merge/group by key column, aggregate numeric cols | column (key), how? (inner/left/right/outer) |
| transform.transpose | transpose DataFrame (swap rows and columns) | (none) |
| transform.drop_rows | drop rows by index range or specific indices | start?, end?, indices? (list) |
| transform.explode | explode comma-separated/list column into separate rows | column, delimiter? |
| transform.cross_join | cartesian product of unique values from two columns | columns (list of 2) |
| transform.stack_columns | stack selected columns into long format (col_name, col_value) | columns? (list) |
| transform.unstack_column | unstack/pivot a column to wide format | index?, column, value? |
| transform.reorder_columns | reorder columns alphabetically or by list | order? (list of col names) |
| transform.duplicate_column | duplicate a column with a new name | column, new_name? |
| transform.flatten_columns | flatten multi-level columns to snake_case | (none) |
| transform.concat_columns | concatenate columns into one | columns, separator?, new_name? |
| transform.split_column | split column by delimiter | column, delimiter? |
| transform.apply_expr | apply math expression to create new column (e.g. "price / area") | expression, new_name? |
| transform.where | replace values where condition is NOT met | column, operator, value, replacement? |
| transform.conditional_column | create if/else column based on condition | column, operator, value, true_value?, false_value?, new_name? |
| transform.row_number | add sequential row number column | new_name? (default _row_number), start? (default 1) |
| transform.chunk_rows | split dataset into fixed-size row chunks, adds _chunk_id column | chunk_size? (default 50) |

### Transform Encode (encode, scale, bin, cast, datetime, sample — output_type MUST be "generate")
| id | what it does | params |
|----|-------------|--------|
| transform.encode_label | label-encode categoricals | column? |
| transform.encode_onehot | one-hot encode | column? |
| transform.encode_binary | encode column as 0/1 based on threshold or specific value | column, threshold?, value? |
| transform.encode_ordinal | ordinal encode with custom order list (or alphabetical) | column, order? (list) |
| transform.scale_standard | z-score standardization | (none) |
| transform.scale_minmax | normalize to [0,1] | (none) |
| transform.scale_robust | RobustScaler (outlier-resistant) | (none) |
| transform.bin_column | bin numeric column into N bins | column, n? (default 5) |
| transform.qcut | quantile-based binning (equal frequency) | column, n? (default 4) |
| transform.round_values | round numeric columns | column?, decimals? (default 2) |
| transform.clip | clip numeric values to min/max bounds | column?, min?, max? |
| transform.winsorize | cap extreme values at percentile bounds | column?, lower? (default 0.05), upper? (default 0.95) |
| transform.normalize_pct | normalize numeric columns to percentages | axis? (columns=row-wise, index=col-wise) |
| transform.pct_change | compute percentage change between consecutive rows | column?, periods? (default 1) |
| transform.cast_columns | cast column(s) to target dtype (int/float/str/bool/category) | column, dtype |
| transform.datetime_parse | parse string column to datetime | column, format? |
| transform.datetime_format | format datetime column to string | column, format? (default %Y-%m-%d), new_name? |
| transform.string_slice | extract substring from text column by position | column, start?, end?, new_name? |
| transform.date_diff | calculate days/months/years between two date columns | column (start), column2 (end), unit? (days/months/years) |
| transform.datetime_decompose | extract year/month/day/weekday/hour/quarter/is_weekend from datetime | column? |
| transform.target_encode_transform | target encoding — replace category with smoothed mean of target | column?, target, smoothing? (default 10) |
| transform.frequency_encode_transform | frequency encoding — replace category with count or proportion | column?, normalize? |
| transform.leave_one_out_encode | leave-one-out target encoding (reduces overfitting) | column?, target |
| transform.inject_null | inject random NaN values | value (number = percentage, e.g. 15 = 15%) |
| transform.sample_rows | random sample of rows | n? |
| transform.head | first N rows | n? |
| transform.tail | last N rows | n? |
| transform.nlargest | top N rows by column value | column?, n? |
| transform.nsmallest | bottom N rows by column value | column?, n? |
| transform.rank | rank values in a column | column?, ascending? |
| transform.cumulative | cumulative sum/count/max/min | column?, agg? (sum/max/min/count) |
| transform.rolling | moving average/sum/std | column?, window? (default 3), agg? (mean/sum/std) |
| transform.shift_column | shift column values up/down by N rows (lag/lead) | column, periods? (default 1) |
| transform.resample | resample time series to different frequency (D/W/M/Q/Y) | column? (date col), freq? (default M), agg? (mean/sum) |
| transform.fill_forward | forward-fill (ffill) nulls only | column? |
| transform.interpolate_values | interpolate missing numeric values (linear/etc) | column?, method? (default linear) |
| transform.shuffle | randomly shuffle all rows | seed? (default 42) |
| transform.train_test_split | split into train/test sets, adds _split column | test_size? (default 0.2), seed?, column? (stratify) |
| transform.stratified_sample | stratified random sampling preserving class ratios | column (stratify by), fraction? (default 0.5), n? |
| transform.systematic_sample | select every N-th row | step? (default 10), offset? (default 0) |

### Viz Basic (common charts — output_type should be "query")
| id | what it does | params |
|----|-------------|--------|
| viz.bar_chart | bar chart of value counts | column?, percentage? |
| viz.histogram | histogram of numeric column | column? |
| viz.scatter | scatter plot | columns? (list: [x, y] or [x, y, color]) |
| viz.line_chart | line chart | column? |
| viz.pie_chart | pie chart (auto-groups >5 cats) | column? |
| viz.count_plot | count/frequency bar | column? |
| viz.heatmap | correlation heatmap | (none) |
| viz.missing_heatmap | missing values pattern | (none) |
| viz.donut_chart | donut chart (pie with hole) | column? |
| viz.stacked_bar | stacked bar (group by 2 categoricals) | columns? (list: [x, color]) |
| viz.grouped_bar | grouped bar chart (2 categoricals side-by-side) | columns? (list: [x, color]) |
| viz.percent_bar | 100% stacked bar chart (proportions) | columns? (list: [x, color]) |
| viz.area_chart | area chart for trends | column? |
| viz.time_series | time series chart | column? |
| viz.bubble_chart | bubble chart (3 numeric cols) | (none) |
| viz.dual_axis | dual Y-axis chart (two numeric cols) | columns? (list: [y1, y2]) |
| viz.step_chart | step chart (discrete steps) | column? |
| viz.lollipop_chart | lollipop chart (dot + stem line) | column? |
| viz.dot_plot | Cleveland dot plot | column? (categorical), value? (numeric) |
| viz.error_bar_chart | bar chart with std error bars | column? (numeric), group? (categorical) |
| viz.cumulative_line | cumulative sum line chart | column? |
| viz.null_bar | null percentage per column bar chart | (none) |
| viz.top_n_bar | top N values bar chart | column?, n? (default 10) |
| viz.correlation_scatter | scatter with trendline and R value | columns? (list: [x, y]) |
| viz.comparison_bar | side-by-side comparison of 2 numeric cols | columns? (list: [a, b]) |
| viz.range_plot | range/band plot (min-max over categories) | column? (numeric), group? (categorical) |
| viz.slope_chart | slope chart comparing two columns (before/after, changes) | column (left), column2 (right), label? |

### Viz Advanced (specialized charts — output_type should be "query")
| id | what it does | params |
|----|-------------|--------|
| viz.box_plot | box plot | column? |
| viz.violin_plot | violin plot | column? |
| viz.pairplot | scatter matrix | (none) |
| viz.distribution | distribution + marginal box | column? |
| viz.qq_plot | QQ plot for normality check | column? |
| viz.density_plot | KDE density plot | column?, group? |
| viz.strip_plot | jitter/strip showing individual points | column?, group? |
| viz.treemap | treemap of categorical column | column? |
| viz.sunburst | sunburst chart | column? |
| viz.parallel_coords | parallel coordinates plot | (none) |
| viz.pareto_chart | pareto chart (bar + cumulative line) | column? |
| viz.waterfall_chart | waterfall chart (incremental changes) | column?, category? |
| viz.funnel_chart | funnel chart (stages/flow) | column? |
| viz.radar_chart | radar/spider chart (normalized means) | (none, uses up to 8 numeric cols) |
| viz.histogram_2d | 2D histogram / density heatmap | columns? (list: [x, y]) |
| viz.ecdf_plot | empirical cumulative distribution function | column? |
| viz.ridgeline | ridgeline/joy plot (overlapping distributions) | column? (numeric), group? (categorical) |
| viz.polar_chart | polar/radial bar chart | column? |
| viz.sankey_chart | Sankey flow diagram between 2 categoricals | columns? (list: [source, target]) |
| viz.candlestick | candlestick/OHLC chart | open?, high?, low?, close? |
| viz.contour_plot | contour density plot of 2 numeric cols | columns? (list: [x, y]) |
| viz.swarm_plot | jitter/swarm plot with better spread | column?, group? |
| viz.marimekko | Marimekko/mosaic chart (variable-width bars) | columns? (list: [x, color]) |
| viz.gauge_chart | single-value gauge chart | column?, agg? (mean/median/sum/min/max) |
| viz.icicle_chart | icicle chart for hierarchical data | columns? (list of categorical cols) |
| viz.density_heatmap | 2D density heatmap between two numeric columns | x?, y?, bins? (default 30) |
| viz.pie_subplots | multiple pie charts side by side for categorical columns | columns? |
| viz.timeline_chart | timeline/Gantt chart from start/end date columns | start?, end?, label? |
| viz.box_comparison | side-by-side box plots comparing distributions | column?, group? |

### Feature Create (create new derived features — output_type "generate")
| id | what it does | params |
|----|-------------|--------|
| feature.polynomial_features | add interaction features | columns? |
| feature.datetime_features | extract year/month/day/dayofweek/hour from datetime | column? |
| feature.ratio_features | create ratio features (col_a/col_b) | columns? |
| feature.cyclical_encode | sin/cos encoding for cyclical features | column, period? |
| feature.lag_features | create lag/shift features for time series | column?, lags? (list or int, default [1,2,3]) |
| feature.text_features | extract text stats: length, word count, digit count, uppercase ratio | column? |
| feature.diff_features | create difference features (first/second order) | column?, periods? (default 1) |
| feature.aggregation_features | group-by stats as new features (mean/std/count) | column? (group-by col), agg_column? |
| feature.interaction_features | multiply pairs of numeric columns | columns? |
| feature.is_null_features | create binary is_null flags for columns with nulls | (none) |
| feature.is_zero_features | create binary is_zero flags for numeric columns | (none) |
| feature.sin_cos_hour | sin/cos encoding for hour-of-day (period=24) | column? |
| feature.is_weekend | weekend flag (1/0) from datetime column | column? |
| feature.is_holiday | basic holiday detection (weekends + common fixed dates) | column? |
| feature.time_since | days since reference date or first date | column?, reference? |
| feature.distance_from_mean | distance from column mean or median | column?, method? (mean/median) |
| feature.feature_cross | cross two categorical columns (A_x_B combinations) | columns? (list of 2) |
| feature.rolling_stats_features | rolling mean/std/min/max as feature columns | column?, window? (default 3) |
| feature.ewm_features | exponentially weighted moving mean/std | column?, span? (default 5) |
| feature.cumulative_features | add cumsum, cummean, cummax, cummin features | column |
| feature.string_length_features | create char_len and word_count from string columns | column? |
| feature.target_encode | mean/target encoding for categoricals | column? (target), encode_column? |
| feature.frequency_encode | encode categoricals by frequency | column? |
| feature.count_encode | encode categoricals by absolute count | column? |
| feature.rare_category_encode | group rare categories into 'Other' | column?, min_count? (default 5) |
| feature.hash_encode | hash encoding for high-cardinality categoricals | column?, n? (features, default 8) |
| feature.ordinal_encode | ordinal encoding with auto-detected alphabetical order | column? |
| feature.label_binarize | multi-label binarization — one column per unique value | column |
| feature.target_binary_encode | encode target as binary (above/below median) | column? (target) |
| feature.woe_encode | Weight of Evidence encoding (requires binary target) | column, target |
| feature.bin_numeric | custom binning with optional labels | column, n? (bins, default 5), labels? |
| feature.rolling_window | rolling window stats (mean/std/min/max) for time series | column?, window? (default 7), stats? (mean,std,min,max) |
| feature.expanding_window | expanding/cumulative window stats for time series | column?, stats? (mean,std,min,max) |
| feature.seasonal_features | cyclical sin/cos encoding for periodic features (month, weekday, hour) | column? |

### Feature Select (select features or transform distributions — output_type "generate")
| id | what it does | params |
|----|-------------|--------|
| feature.feature_importance | feature importance ranking | column? (target) |
| feature.pca | principal component analysis | n? (components) |
| feature.correlation_filter | drop highly correlated features | value? (threshold, default 0.95) |
| feature.variance_filter | drop low-variance features | value? (threshold, default 0.01) |
| feature.select_k_best | select top K features by statistical test | column? (target), k? (default 10) |
| feature.mutual_info | mutual information scores (numeric + categorical) | column? (target) |
| feature.auto_feature_select | auto-select: drop low-variance + high-correlation | var_threshold? (default 0.01), corr_threshold? (default 0.95) |
| feature.rfe_select | Recursive Feature Elimination — iteratively remove weakest features | column (target), n_features? |
| feature.mutual_info_select | rank features by mutual information with target, drop low-scoring | column (target), threshold? (default 0.01), top_k? |
| feature.variance_threshold | drop features with variance below threshold (near-constant) | threshold? (default 0.01) |
| feature.correlation_select | drop one of each pair of highly correlated features | threshold? (default 0.95), target? |
| feature.boruta_select | Boruta all-relevant feature selection (shadow features + RF) | column (target), max_iter? (default 50) |
| feature.log_transform | log transform skewed numeric cols | column? |
| feature.power_transform | Box-Cox/Yeo-Johnson normalization | column?, method? (yeo-johnson/box-cox) |
| feature.sqrt_transform | square root transform for moderate skew | column? |
| feature.quantile_transform | map values to uniform/normal distribution | column?, distribution? (normal/uniform) |
| feature.log1p_transform | log(1+x) transform for skewed non-negative data | column? |
| feature.abs_transform | absolute value transform | column? |
| feature.reciprocal_transform | 1/x reciprocal transform (zeros become NaN) | column? |
| feature.rank_transform | percent rank (0-1) for numeric columns | column? |
| feature.kbins_discretize | KBins discretizer (uniform/quantile/kmeans) | column?, n? (bins, default 5), strategy? (quantile/uniform/kmeans) |
| feature.exponential_transform | exponential transform (e^x) | column? |
| feature.clip_features | clip values to percentile range | column?, lower? (default 0.01), upper? (default 0.99) |
| feature.zscore_features | add z-score columns for numeric features | column? |
| feature.boxcox_transform | Box-Cox transform (positive values only) | column? |
| feature.yeo_johnson_transform | Yeo-Johnson transform (handles negatives) | column? |
| feature.winsorize | cap extreme values at percentile bounds | column?, lower? (default 0.05), upper? (default 0.95) |
| feature.svd_features | SVD dimensionality reduction (works with sparse data) | n_components? (default 3) |
| feature.percentile_rank | add percentile rank (0-100) for numeric column(s) | column? |

### NLP Process (text cleaning, normalization, preprocessing — output_type "generate")
| id | what it does | params |
|----|-------------|--------|
| nlp.text_clean | clean text: lowercase, remove HTML/URLs/emails/punctuation/numbers, normalize whitespace | column?, strategy? (all/lowercase/no_punct/no_numbers/no_html/no_urls/no_emails) |
| nlp.remove_stopwords | remove English stopwords from text columns | column?, extra_words? (list) |
| nlp.tokenize | split text into tokens, create token_count column | column? |
| nlp.text_normalize | normalize: strip accents + basic stemming + lowercase | column?, stem? (default true) |
| nlp.text_encode | encode text as integer sequences (word→ID) for deep learning | column?, max_vocab? (default 5000), max_len? (default 100) |
| nlp.text_dedup | find/remove near-duplicate texts using TF-IDF cosine similarity | column?, threshold? (default 0.9), action? (flag/remove) |
| nlp.text_dedup_exact | fast exact-match text deduplication (case-insensitive) | column?, keep? (first/last) |
| nlp.text_mask_pii | mask PII: emails, phones, credit cards, SSN, IPs, URLs → [EMAIL] etc. | column? |
| nlp.text_augment | text augmentation: random word delete/swap to expand dataset | column?, n? (copies, default 1), strategy? (delete/swap/mixed) |
| nlp.text_filter | filter rows by text criteria: min/max length, contains, min words | column?, min_len?, max_len?, min_words?, contains?, not_contains? |
| nlp.text_chunk | split long texts into fixed-size word chunks (new rows per chunk) | column?, chunk_size? (default 200), overlap? (default 20) |
| nlp.text_concat | combine multiple text columns into one corpus column | columns? (list), separator? (default " "), new_name? (default "text_combined") |
| nlp.text_replace | find and replace text patterns (regex or literal) | column?, pattern, replacement, mapping? (dict), regex? (default true) |
| nlp.text_split_sentences | split text into individual sentences (new rows per sentence) | column? |
| nlp.text_oversample | oversample minority text classes to balance dataset | column? (label col) |
| nlp.text_truncate_pad | truncate or pad text to fixed word count | column?, max_words? (default 128), pad_token? |
| nlp.text_to_paragraphs | split text by blank lines into paragraphs (new rows) | column? |
| nlp.text_remove_rare | remove words appearing below frequency threshold | column?, min_freq? (default 2) |
| nlp.text_window | extract sliding window contexts around a keyword | column?, keyword, window? (default 5) |
| nlp.text_label_rules | create labels from keyword rules: {label: [keywords]} | column?, mapping (dict), default? |
| nlp.translate | translate text column to another language (Google Translate) | column?, source? (default auto), target? (default en) |
| nlp.text_uppercase | convert text column to UPPERCASE | column? |
| nlp.text_lowercase | convert text column to lowercase | column? |
| nlp.text_title_case | convert text column to Title Case | column? |
| nlp.text_pattern_match | filter rows where text matches regex pattern | column?, pattern |

### NLP Extract (text analysis, features, vectorization — mixed output_type)
| id | what it does | params |
|----|-------------|--------|
| nlp.tfidf | TF-IDF vectorization → N feature columns | column?, n? (max features, default 50) |
| nlp.bow | Bag of Words count vectorization → N feature columns | column?, n? (max features, default 50) |
| nlp.ngrams | extract word n-gram features via TF-IDF | column?, n? (gram size, default 2), max_features? |
| nlp.regex_extract | extract patterns: email/url/hashtag/mention/phone/number/custom | column?, pattern? (email/url/hashtag/mention/phone/number/all), regex? |
| nlp.sentiment_score | lexicon-based sentiment scoring (positive/negative/compound) + chart | column? |
| nlp.word_frequency | ★ CORPUS-WIDE top-N most frequent words ACROSS ALL ROWS combined + bar chart (use for "top keywords in data", "most common words") | column?, n? (default 20), remove_stopwords? |
| nlp.text_similarity | cosine similarity matrix using TF-IDF + heatmap (output_type=query) | column? |
| nlp.vocab_stats | vocabulary statistics: unique tokens, TTR, avg word length, hapax (output_type=query) | column? |
| nlp.language_detect | detect language per row from Unicode character ranges + pie chart | column? |
| nlp.hash_vectorize | feature hashing — fast, memory-efficient text vectorization | column?, n? (features, default 32) |
| nlp.keyword_extract | extract top-N keywords PER EACH ROW individually (NOT corpus-wide) using TF-IDF | column?, n? (keywords per doc, default 5) |
| nlp.char_features | character-level features: punct/digit/upper/lower/space ratios, avg word length | column? |
| nlp.sentence_features | sentence-level stats: count, avg/min/max length, question/exclamation counts | column? |
| nlp.readability_score | readability metrics: Flesch Reading Ease, Coleman-Liau, ARI | column? |
| nlp.emoji_features | extract emoji + emoticon count, ratio, and list per row | column? |
| nlp.collocations | find significant word pair collocations ranked by PMI + chart (output_type=query) | column?, n? (default 20) |
| nlp.word_cloud | word frequency treemap visualization (output_type=query) | column?, n? (words, default 40) |
| nlp.class_balance_text | analyze text label class balance + distribution chart (output_type=query) | column (label col), text_column? |
| nlp.spelling_features | OOV/spelling quality features: rare-word count and ratio per row | column? |
| nlp.doc_term_matrix | build document-term frequency matrix (full vocab) | column?, n? (max features, default 100) |
| nlp.word_overlap | Jaccard word overlap between two columns or consecutive rows | columns? (list of 2), column? |
| nlp.text_length_dist | analyze text length distribution with histogram (output_type=query) | column? |
| nlp.text_unique_words | extract words unique to each document (corpus-level rarity) | column? |
| nlp.text_count_pattern | count occurrences of a pattern per row, optionally filter | column?, pattern, filter? (bool) |
| nlp.text_summary_report | comprehensive text dataset report: stats, quality, recommendations (output_type=query) | column? |
| nlp.text_stratified_sample | stratified random sample maintaining label distribution | column? (label col), n? (default 100) |
| nlp.text_ngram_frequency | word n-gram frequency analysis with bar chart (output_type=query) | column?, n? (gram size, default 2), top? (default 20) |
| nlp.text_extract_numbers | extract all numbers from text into a new column | column? |
| nlp.text_pos_patterns | detect POS-like surface patterns: all_caps/capitalized/numeric ratios | column? |
| nlp.text_diversity_index | Simpson diversity index of words per document | column? |
| nlp.named_entity_extract | extract entities (email/url/phone/number/hashtag) from text | column?, entity? (default all) |

### Analysis Explore (EDA, profiling, quality, trends, comparisons)
| id | what it does | output | params |
|----|-------------|--------|--------|
| analysis.compare_extremes | compare rows with max vs min value — use for "ถูกสุดกับแพงสุดต่างกันยังไง/cheapest vs most expensive" | table+chart | column? (numeric col to compare by) |
| analysis.deep_profile | deep statistical profile of one column: distribution, outliers, quartiles | table+chart | column? |
| analysis.group_insights | compare stats across groups — use for "แต่ละกลุ่มเป็นยังไง/compare by category" | table+chart | column? (group col), value_column? (numeric) |
| analysis.data_quality | comprehensive data quality scores per column + overall score | table+chart | (none) |
| analysis.correlation_insights | find top correlations with explanation + scatter plots | table+chart | n? (default 10) |
| analysis.compare_columns | side-by-side comparison of two columns: stats + histogram | table+chart | columns (list of 2) |
| analysis.trend_detect | detect trends: direction, slope, moving average | table+chart | column?, window? (default 5) |
| analysis.segment_analysis | auto-segment into quantile groups and describe each | table | column?, n? (segments, default 4) |
| analysis.auto_eda | automated EDA: key findings, quality issues, recommendations | table | (none) |
| analysis.distribution_analysis | distribution shape analysis (skew, kurtosis, normality) per numeric col | (none) |
| analysis.missing_value_analysis | deep missing value pattern analysis (co-occurrence, MCAR hint) | (none) |
| analysis.categorical_analysis | deep analysis of all categorical columns (cardinality, mode, entropy) | (none) |
| analysis.numeric_summary | comprehensive numeric columns summary in one table | (none) |
| analysis.top_n_analysis | analyze top N rows by a metric with details | column?, n? (default 10) |
| analysis.bottom_n_analysis | analyze bottom N rows by a metric | column?, n? (default 10) |
| analysis.percentile_analysis | compare stats across percentile bands (Q1-Q4) | column? |
| analysis.variance_analysis | variance contribution per feature (% of total variance) | (none) |
| analysis.change_point_detect | detect change points in numeric series (sliding window mean diff) | column?, window? (default 10) |
| analysis.target_analysis | analyze relationship between each feature and a target column | column? (target) |
| analysis.pareto_analysis | Pareto 80/20 rule analysis on a column | column? |
| analysis.data_completeness | data completeness scorecard per column + overall score | (none) |
| analysis.seasonality_detect | detect seasonality via autocorrelation + chart | column? |
| analysis.gap_analysis | find significant gaps in sorted numeric data | column? |
| analysis.benchmark_compare | compare column means against benchmark targets | benchmarks? (dict) |
| analysis.sensitivity_analysis | sensitivity of target to each feature | column? (target) |
| analysis.correlation_network | correlation edges above threshold | threshold? (default 0.5) |
| analysis.feature_drift | compare first/second half distributions | (none) |
| analysis.sample_bias_check | check sampling bias vs full population | sample_frac? (default 0.3) |
| analysis.concentration_analysis | Lorenz curve + Gini coefficient | column? |
| analysis.diminishing_returns | detect diminishing returns between two columns | columns? (list of 2) |
| analysis.categorical_target_crosstab | crosstab heatmap of two categorical columns | column? (target), feature_column? |
| analysis.data_readiness_score | overall ML data readiness score (completeness, balance, size) | (none) |
| analysis.data_profiling_report | comprehensive column-by-column data profiling | (none) |
| analysis.imbalance_report | class distribution analysis with imbalance ratio, severity, and resampling recommendation | column (target)? |
| analysis.schema_validate | validate column types, null constraints, and value ranges against expected schema | types? (dict), not_null? (list), ranges? (dict) |
| analysis.constraint_check | check uniqueness, value range, regex format, and allowed values constraints | unique? (list), ranges? (dict), patterns? (dict), allowed_values? (dict) |

### Analysis Model (clustering, ML, hypothesis tests, time series, drift)
| id | what it does | params |
|----|-------------|--------|
| analysis.anomaly_detect | detect anomalies using IQR/Z-score — flag outlier rows + scatter chart | column?, method? (iqr/zscore) |
| analysis.cluster_kmeans | K-Means clustering with auto k selection (silhouette) + 2D scatter | max_k? (default 8) |
| analysis.pca_2d | PCA 2D projection with explained variance + scatter | (none) |
| analysis.outlier_isolation_forest | Isolation Forest anomaly detection + scatter | contamination? (default 0.05) |
| analysis.feature_selection_auto | automatic feature selection (variance + correlation + mutual info) | column? (target) |
| analysis.hypothesis_test | auto choose t-test/Mann-Whitney based on normality | column? (numeric), group_column? (categorical) |
| analysis.regression_quick | quick OLS linear regression + scatter + R-squared + coefficients | column? (target), feature? |
| analysis.multicollinearity_check | VIF-based multicollinearity detection | (none) |
| analysis.ab_test | A/B test with significance (p-value, effect size, confidence interval) | column? (metric), group_column? |
| analysis.cluster_profile | profile each cluster's characteristics after K-Means | n? (clusters, default 3) |
| analysis.feature_interaction | detect feature interaction effects on target | column? (target) |
| analysis.cohort_analysis | cohort-based analysis by categorical groups | column? (group), value_column? |
| analysis.rfm_analysis | RFM scoring on first 3 numeric columns | (none) |
| analysis.effect_size | Cohen's d effect sizes between two groups | column? (group col) |
| analysis.bootstrap_ci | bootstrap confidence intervals for column mean | column?, n_bootstrap?, confidence? |
| analysis.cross_correlation | cross-correlation between two columns at various lags | columns? (list of 2) |
| analysis.survival_curve | basic survival/retention curve | column? |
| analysis.prediction_baseline | compute naive prediction baselines (mean/mode) | column? (target) |
| analysis.time_series_decompose | decompose time series into trend + seasonal + residual | column?, period? |
| analysis.lift_analysis | lift/gain analysis for score column vs binary target | column? (score), target? |
| analysis.market_basket | association rule mining (support, confidence, lift) | group? (transaction), column? (item), min_support? |
| analysis.granger_causality | Granger causality test between two time-ordered columns | column? (x), y?, max_lag? (default 4) |
| analysis.dbscan_clustering | DBSCAN density-based clustering (finds arbitrary-shape clusters + noise) | eps? (default 0.5), min_samples? (default 5) |
| analysis.hierarchical_clustering | agglomerative hierarchical clustering with dendrogram | n_clusters? (default 3), method? (ward/complete/average) |
| analysis.stationarity_test | Augmented Dickey-Fuller test for time series stationarity | column? |
| analysis.data_drift_detect | compare two halves of dataset for distribution drift using KS test and PSI | column?, split? |
"""

PLANNER_PROMPT = """\
You are the planner for a data-science agent. Given a user request and dataset,
output a JSON execution plan.

## CONVERSATION HISTORY (recent messages for context)
{conversation_history}

## CURRENT USER REQUEST
{user_message}

## DATASET
{df_context}

## AVAILABLE HANDLERS (use whenever possible — instant, no code generation)
{handler_catalog}

## DECISION RULES — READ CAREFULLY

### FIRST: Is this about the dataset?
Before anything else, ask yourself: **Does this message require the dataset to answer?**
- If the user asks a general knowledge question, casual chat, opinion, math, coding help,
  or anything NOT requiring the loaded data → set `"direct_answer": true` with empty steps.
- If the user references columns, data, statistics, charts, cleaning, or anything that
  needs the actual dataset → proceed to handler/codegen below.

Examples of direct_answer (NOT about the dataset):
- "ไก่ ไข่ ไก่ หมา คำไหนมีมากที่สุด" → general question, answer directly
- "what is machine learning?" → general knowledge
- "how do I write a for loop in Python?" → coding help
- "thank you" / "ok" / "got it" → casual chat
- "1+1=?" → simple math
- "what's the weather today?" → off-topic

Examples of dataset-related (use handler/codegen):
- "show nulls" → stats.null_report (references data quality)
- "plot price distribution" → viz (references column)
- "clean the data" → clean handlers (modifies dataset)
- "how many rows?" → stats.shape (about the dataset)
- "คอลัมน์ไหนมี null มากที่สุด" → stats.null_report (about columns)

### When to use a handler
ALWAYS prefer a handler when one fits. Handlers are instant, reliable, and tested.
Scan the handler table above — if ANY handler matches the user's intent, use it.

### When to use codegen
Use codegen ONLY when NO handler in the table above can do it:
  - **Binning/cutting** — "แบ่งเป็น N ระดับ", "split into N levels/bins/ranges" → ALWAYS codegen with pd.cut, never viz.bar_chart
  - Custom calculations (percentages, ratios, derived metrics)
  - Pivot, melt, reshape, merge, join
  - Moving average, rolling window, cumulative operations
  - Complex multi-condition filtering
  - Any custom math/logic not covered by a handler
IMPORTANT: if user says "แบ่ง/split/divide into N levels/bins" → this is a CODEGEN task (pd.cut), NOT a handler task

### Multi-dataset: which dataframe does each step run on?
The primary dataframe is `df`. Any "Additional datasets available as variables" listed
above are also loaded and available by their variable name.

For EACH step you emit, you MAY add an optional field `"dataset": "<variable_name>"`
to tell the executor which dataframe that step targets.
  - Omitted or `"df"` → run on the primary dataframe (default).
  - Equal to one of the extra variable names → run on that extra dataframe instead.

NEVER emit two handler steps without `dataset` set when the user mentioned a non-primary
dataset — step 2 would otherwise just run on step 1's output instead of the intended df.

### CRITICAL CONTRACT: only ONE result table is shown to the user
The chat surfaces ONE table per response — the result of the LAST step that produces a
DataFrame. So plan accordingly:

- For ANY comparison / diff / merge / contrast question across datasets (words like
  "compare", "vs", "difference", "ต่างกัน", "เทียบ", "ระหว่าง"), emit exactly ONE codegen
  step that builds a single combined DataFrame containing rows/columns from each dataset
  side by side. Do NOT split into per-dataset handler steps — the user will only see the
  last one and miss the comparison.

- For "show X for each of A and B" questions, you may still use per-dataset handler
  steps, but then add ONE final codegen step whose result_df concatenates the per-step
  outputs into a single labeled table. Otherwise the user only sees the last dataset.

- A codegen step that only prints scalars (e.g. "the difference is 1053") with no
  result_df leaves the last handler's table as the user-facing artifact — that table
  almost certainly belongs to a different dataset than what the user is asking about,
  which is confusing. Always have the final step produce result_df.

### output_type
  - "query" → stats, viz, questions, any read-only operation
  - "generate" → cleaning, transforms, data generation — anything that creates/modifies data

### When to create a chart vs NOT — THINK before adding any visualization
**Only create a chart when (1) the user explicitly asks AND (2) the data is actually
plottable. Either condition failing means NO viz step. An empty axis with no bars is
worse than no chart at all.**

DO create a chart when the user says:
- "show", "plot", "chart", "graph", "visualize", "แสดงกราฟ", "พล็อต", "ดูเป็นภาพ"
- "distribution of X", "breakdown of X" (implicitly visual)

Do NOT create a chart when the user asks:
- A question with a short answer: "what is the average price?" → just compute + answer in text
- A comparison: "cheapest vs most expensive?" → use stats/analysis handlers, return text+table
- A count: "how many rows have null?" → stats handler, text answer
- "describe" / "summary" / "info" → stats handler, no chart needed
- ANY yes/no question: "is there correlation?" → stats.correlation → text answer

**Rule: if the answer can be a number, a sentence, or a small table — do NOT add a chart step.**

### Data-suitability check — MANDATORY before adding any viz step
Even when the user asks for a chart, INSPECT the DATASET section above and SKIP the
viz step if any of these are true (return text/table only):

- **No numeric or low-cardinality categorical column exists.** A column is "low-cardinality
  categorical" only if its unique count is ≤ ~30 AND its values are short tokens, not prose.
- **The column you would plot is freeform text** — descriptions, code, markdown, schemas,
  multi-sentence content, or anything where the typical cell length exceeds ~30 characters.
  Plotting "value counts" of such a column gives N bars of height 1 — useless.
- **Cardinality equals row count** for the plotted column (every row unique) — bar/pie/count
  would show only height-1 bars and convey nothing.
- **All-null or constant column** — nothing varies to visualize.
- **Fewer than ~3 distinct values to show** — a 1-bar or 2-bar chart is noise, not signal.
- **The whole dataset is project notes, task lists, documentation, schemas, or prose** — no
  meaningful axis exists. Respond with a table or text summary only.

A chart that would render with empty axes or single-height bars is a FAILURE worse than
omitting it. When unsure → omit and explain in text. The user can ask for a specific
visualization on a specific column afterwards.

### Chart selection — when a chart IS needed
When the user explicitly asks for a visualization, REASON about:
1. **What question is the user asking?** — count, composition, distribution, relationship, trend, comparison?
2. **What does the DATASET section say about the column?** — check dtype, unique count, is it numeric or categorical?
3. **Pick the chart that BEST answers the question:**
   - User wants to see counts/frequency of categories → bar_chart or count_plot
   - User wants percentages/proportions/share (parts of whole) → pie_chart
   - User wants to understand how numeric values are spread → histogram or distribution
   - User wants to compare a numeric value across groups → box_plot or violin_plot
   - User wants to see relationship between 2 numeric columns → scatter
   - User wants to see trends over time/sequence → line_chart
   - User wants a correlation overview → heatmap
4. **Do NOT default to bar_chart or pie_chart blindly.** Read the user's actual intent word by word.

### NLP / text data — THINK before picking handlers
When the user works with text/NLP data or asks to prepare text for ML:
1. **Text cleaning pipeline** (typical order): text_clean → remove_stopwords → text_normalize → then vectorize
2. **Vectorization choices**:
   - Simple classification → nlp.tfidf or nlp.bow (fast, interpretable)
   - High-cardinality text → nlp.hash_vectorize (memory-efficient)
   - Deep learning input → nlp.text_encode (integer sequences)
   - Capture phrases → nlp.ngrams (bigrams/trigrams)
3. **Analysis** (read-only, query): nlp.word_frequency, nlp.vocab_stats, nlp.text_similarity, nlp.collocations, nlp.word_cloud, nlp.class_balance_text
4. **Feature extraction** (generate): nlp.sentiment_score, nlp.regex_extract, nlp.language_detect, nlp.char_features, nlp.sentence_features, nlp.readability_score, nlp.emoji_features, nlp.keyword_extract, nlp.spelling_features
5. **IMPORTANT — keyword/word frequency distinction**:
   - "top keywords across all rows" / "คำที่เยอะที่สุดรวมทุก rows" → **nlp.word_frequency** (corpus-wide aggregate)
   - "keywords per each row" / "keyword ของแต่ละ row" → nlp.keyword_extract (per-document)
   - "word cloud" / "show common words" → nlp.word_cloud (corpus-wide treemap)
   - If user says "รวมทุก rows" or "across all" or "ทั้งหมด" → ALWAYS use nlp.word_frequency, NOT keyword_extract
5. **Data prep**: nlp.text_dedup (remove duplicates), nlp.text_filter (remove short/empty), nlp.text_mask_pii (anonymize), nlp.text_chunk (split long docs), nlp.text_augment (expand small datasets), nlp.text_concat (merge text columns)
6. Combine multiple NLP steps when the user says "prepare text" or "preprocess for NLP"

### Smart analysis — use analysis handlers for complex questions
When the user asks complex analytical questions, prefer analysis handlers over codegen:
- "compare max vs min" / "เปรียบเทียบสูงสุดกับต่ำสุด" → analysis.compare_extremes
- "tell me about column X in detail" / "อธิบาย column X" → analysis.deep_profile
- "compare groups" / "เปรียบเทียบกลุ่ม" → analysis.group_insights
- "find outliers" / "หาค่าผิดปกติ" → analysis.anomaly_detect
- "check data quality" / "ตรวจคุณภาพ" → analysis.data_quality
- "what correlates with X?" / "อะไรสัมพันธ์กับ X" → analysis.correlation_insights
- "compare column A vs B" → analysis.compare_columns
- "is there a trend?" / "มี trend ไหม" → analysis.trend_detect
- "segment the data" / "แบ่งกลุ่มข้อมูล" → analysis.segment_analysis
- "analyze this data" / "วิเคราะห์ข้อมูล" / "EDA" → analysis.auto_eda

### Follow-up questions — USE CONVERSATION HISTORY
When the user says "plot that", "show me", "ขอแบบ plot", "ทำเป็น chart", "เดิม", "อันเดิม"
or refers to something from a previous message:
1. Read the CONVERSATION HISTORY section above
2. Identify which column/topic was discussed previously
3. Use THAT column — do NOT pick a random default column
Example: if history says "parking value counts" and user says "plot it" → use column="parking"

### Percentage / proportion questions — ALWAYS use stats.value_counts
When the user asks "how many percent?", "percentage", "proportion", "สัดส่วน", "เปอร์เซ็นต์", "กี่เปอร์เซ็น":
- Use `stats.value_counts` with the relevant column → it automatically calculates count AND percentage
- Do NOT use codegen for simple percentage calculations
- Example: "ห้องนอนแต่ละแบบกี่เปอร์เซ็นต์" → stats.value_counts with column="bedrooms"
- stats.value_counts already returns: column, count, percentage (count/total*100)

### Other rules
1. Column names MUST match actual columns from DATASET section — NEVER invent column names.
2. Use as many steps as needed to fully answer the request. 1 step for simple tasks, more for complex ones — no upper limit.
3. Each step has EITHER "handler" OR "codegen" — never both.
4. For "tell me about the dataset" / "overview" / "info" / "describe" / "summary" → stats.describe
5. For "show nulls" / "missing values" / "how many nulls" → stats.null_report
6. For "fill nulls" / "fill missing" → clean.fill_nulls (NOT stats.null_report)
7. For "inject/create/generate nulls" → transform.inject_null
8. For any chart request → use the matching viz handler (apply smart chart selection above)
9. For "correlation" (no chart word) → stats.correlation
10. For "correlation heatmap" → viz.heatmap
11. If user speaks Thai, understand the intent and plan normally — do NOT translate.
12. For multi-step requests ("clean then show chart"), create multiple steps spanning categories.
13. For "percentage" / "เปอร์เซ็นต์" / "สัดส่วน" of a column → stats.value_counts (already includes percentage)
14. **NO duplicate chart steps** — if a codegen step already creates a chart (px.bar, px.scatter, go.Figure, px.pie, etc.), do NOT add a separate viz handler step after it. One chart per request is enough.
15. For binning + plot requests ("แบ่งเป็น N ระดับ แล้ว plot") → use ONE codegen step that does both pd.cut AND creates the chart. Do NOT split into codegen + viz.bar_chart.

## OUTPUT FORMAT — valid JSON, no markdown fences, no explanation

For handler steps:
{{"step_num":1,"description":"...","handler":{{"id":"category.sub","params":{{}}}}}}

For codegen steps:
{{"step_num":2,"description":"...","codegen":{{"task":"detailed Python task description","produces":"dataframe|chart|text"}}}}

Full format (data-related):
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

User: "คิดเป็นอย่างละกี่เปอร์เซ็นต์" (follow-up to a column count)
{{"understanding":"Calculate percentage for each category","output_type":"query","steps":[{{"step_num":1,"description":"Value counts with percentage","handler":{{"id":"stats.value_counts","params":{{"column":"bedrooms"}}}}}}]}}

User: "correlation heatmap"
{{"understanding":"Show correlation heatmap","output_type":"query","steps":[{{"step_num":1,"description":"Correlation heatmap","handler":{{"id":"viz.heatmap","params":{{}}}}}}]}}

User: "inject 15% random nulls"
{{"understanding":"Inject random null values","output_type":"generate","steps":[{{"step_num":1,"description":"Inject 15% random nulls","handler":{{"id":"transform.inject_null","params":{{"value":15}}}}}}]}}

User: "remove duplicates then fill missing values"
{{"understanding":"Clean: deduplicate then fill nulls","output_type":"generate","steps":[{{"step_num":1,"description":"Remove duplicates","handler":{{"id":"clean.remove_duplicates","params":{{}}}}}},{{"step_num":2,"description":"Fill missing values","handler":{{"id":"clean.fill_nulls","params":{{"strategy":"auto"}}}}}}]}}

User: "split price into 5 levels and show percentage"
{{"understanding":"Bin price into 5 ranges with percentages","output_type":"query","steps":[{{"step_num":1,"description":"Bin price into 5 ranges with count and percentage","codegen":{{"task":"Use pd.cut on the price column with bins=5. Count each bin, calculate percentage. Create result DataFrame with Range/Count/Percentage. Format bin labels as human-readable (34K-154K). Create bar chart: fig = px.bar(result, x='Range', y='Count', title='Price Distribution', text='Count')","produces":"dataframe"}}}}]}}

User: "แบ่งราคาบ้านเป็น 6 ระดับและแสดงว่าแต่ละระดับมีจำนวนเท่าไหร่ plot ให้ดูได้ไหม"
{{"understanding":"Bin price into 6 ranges and plot bar chart","output_type":"query","steps":[{{"step_num":1,"description":"Bin price into 6 ranges and plot","codegen":{{"task":"Use pd.cut(df['price'], bins=6) to create price ranges. Count each range. Create result DataFrame with columns Range and Count. Format range labels as human-readable numbers. Create bar chart: fig = px.bar(result, x='Range', y='Count', title='Price by Range (6 levels)', text='Count', color='Count', color_continuous_scale=['#FFE0CC','#E8500A'])","produces":"dataframe"}}}}]}}

User: "describe the dataset"
{{"understanding":"Show descriptive statistics","output_type":"query","steps":[{{"step_num":1,"description":"Descriptive statistics","handler":{{"id":"stats.describe","params":{{}}}}}}]}}

User: "remove duplicates, fill nulls, then show a bar chart of bedrooms" (multi-step, 3 categories)
{{"understanding":"Clean dataset then visualize","output_type":"generate","steps":[{{"step_num":1,"description":"Remove duplicates","handler":{{"id":"clean.remove_duplicates","params":{{}}}}}},{{"step_num":2,"description":"Fill missing values","handler":{{"id":"clean.fill_nulls","params":{{"strategy":"auto"}}}}}},{{"step_num":3,"description":"Bar chart of bedrooms","handler":{{"id":"viz.bar_chart","params":{{"column":"bedrooms"}}}}}}]}}

User: "SMOTE oversample then select features with Boruta"
{{"understanding":"Balance classes then select relevant features","output_type":"generate","steps":[{{"step_num":1,"description":"SMOTE oversampling","handler":{{"id":"clean.smote_oversample","params":{{"column":"target"}}}}}},{{"step_num":2,"description":"Boruta feature selection","handler":{{"id":"feature.boruta_select","params":{{"column":"target"}}}}}}]}}

IMPORTANT: Output ONLY valid JSON. No markdown, no explanation, no code fences.
"""


# ── Stage 1: Sub-category router ────────────────────────────────────────────

ROUTER_PROMPT = """\
You are a router for a data-science agent. A dataset is loaded with columns: {columns}

Recent conversation:
{history_summary}

Current user message: "{user_message}"

Your job: classify into 1-3 handler categories.
Use conversation history to understand follow-up questions.

Categories (pick from these 7):
- stats: describe, shape, nulls, value counts, correlation, hypothesis tests, chi2, ANOVA, t-test, compare, min/max, percentage, proportion, class balance
- clean: fill nulls, remove duplicates, drop columns, fix types, SMOTE, oversample, undersample, clip outliers, rename, anonymize
- transform: filter, sort, group, pivot, merge, encode, scale, bin, cast dtype, datetime, sample, split, reshape
- viz: bar, pie, line, scatter, histogram, heatmap, box, violin, treemap, sankey, waterfall, radar, distribution, any chart/plot/graph
- feature: create features (lag, rolling, interaction, polynomial), select features (PCA, RFE, Boruta, mutual info, variance)
- nlp: text clean, tokenize, stopwords, translate, TF-IDF, sentiment, NER, word frequency, ngrams, keyword extract
- analysis: EDA, data quality, profiling, trends, comparisons, clustering, anomaly detection, drift, segment, outlier analysis

IMPORTANT — direct_answer rules:
- direct_answer=true ONLY when the question has ZERO connection to the loaded dataset
- Examples: "what is Python?" → direct_answer. "who made you?" → direct_answer
- If the question could be answered by looking at the data → NOT direct_answer
- "บ้านราคาถูกสุดกับแพงที่สุดต่างกันยังไง" → stats, analysis (about the data!)
- "คิดเป็นเปอร์เซ็นต์" → stats (percentage of data values)
- When in doubt with dataset loaded → route to handlers, never direct_answer

Output ONLY JSON: {{"categories": ["cat1"], "direct_answer": false}}
Or: {{"categories": [], "direct_answer": true}}"""


def _split_catalog_by_category() -> dict[str, str]:
    """Split HANDLER_CATALOG into 7 broad categories (merging sub-categories).

    Catalog headers like "### Stats Summary" and "### Stats Test" both map to "stats".
    This gives the router fewer, clearer choices while keeping all handlers accessible.
    """
    # Map catalog sub-headers → 7 broad categories
    header_to_cat = {
        "Stats Summary": "stats",
        "Stats Test": "stats",
        "Clean Values": "clean",
        "Clean Structure": "clean",
        "Transform Reshape": "transform",
        "Transform Encode": "transform",
        "Viz Basic": "viz",
        "Viz Advanced": "viz",
        "Feature Create": "feature",
        "Feature Select": "feature",
        "NLP Process": "nlp",
        "NLP Extract": "nlp",
        "Analysis Explore": "analysis",
        "Analysis Model": "analysis",
    }

    sections: dict[str, list[str]] = {}
    current_key = ""
    current_lines: list[str] = []

    for line in HANDLER_CATALOG.split("\n"):
        if line.startswith("### "):
            if current_key and current_lines:
                sections.setdefault(current_key, []).extend(current_lines)
            header = line[4:].strip()
            current_key = ""
            for prefix, key in sorted(header_to_cat.items(), key=lambda x: -len(x[0])):
                if header.startswith(prefix):
                    current_key = key
                    break
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_key and current_lines:
        sections.setdefault(current_key, []).extend(current_lines)

    return {k: "\n".join(v) for k, v in sections.items()}


_CATALOG_SECTIONS: dict[str, str] = {}


def _get_catalog_sections() -> dict[str, str]:
    """Lazy-init catalog sections."""
    global _CATALOG_SECTIONS
    if not _CATALOG_SECTIONS:
        _CATALOG_SECTIONS = _split_catalog_by_category()
    return _CATALOG_SECTIONS


def _verify_direct_answer(
    user_message: str, columns: list[str], model_id: str | None,
) -> bool:
    """AI verification: is this question truly unrelated to the loaded dataset?

    Called only when router says direct_answer=true but data is loaded.
    Returns True if the question IS about the data (should NOT be direct_answer).
    Uses max_tokens=10 — expects just YES or NO.
    """
    from api.llm import get_llm

    prompt = (
        f'Dataset columns: {", ".join(columns[:15])}\n'
        f'Question: "{user_message}"\n\n'
        f"Could this question be answered by looking at or analyzing the dataset? YES or NO"
    )
    try:
        llm = get_llm(temperature=0.0, max_tokens=10, model_id=model_id)
        answer = llm.invoke(prompt).content.strip().upper()
        is_about_data = answer.startswith("YES")
        log.info("  verify direct_answer → question about data? %s", is_about_data)
        return is_about_data
    except Exception as e:
        log.warning("  verify direct_answer failed (%s) → assuming about data", e)
        return True


def _route_categories(
    user_message: str, columns: list[str], model_id: str | None,
    history: list | None = None,
) -> tuple[list[str], bool]:
    """Stage 1: Lightweight LLM call to classify intent into sub-categories."""
    from api.llm import get_llm

    # Build short history summary for router context (last 3 messages, 150 chars each)
    history_lines = []
    if history:
        for msg in history[-3:]:
            role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if len(content) > 150:
                content = content[:150] + "..."
            history_lines.append(f"{role}: {content}")
    history_summary = "\n".join(history_lines) if history_lines else "(no previous messages)"

    prompt = ROUTER_PROMPT.format(
        user_message=user_message,
        columns=", ".join(columns[:15]),
        history_summary=history_summary,
    )

    try:
        router_llm = get_llm(temperature=0.0, max_tokens=200, model_id=model_id)
        response = router_llm.invoke(prompt)
        raw = response.content.strip()
        if "```" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip() if "```json" in raw else raw.split("```")[1].split("```")[0].strip()

        result = json.loads(raw)
        categories = result.get("categories", [])
        direct = result.get("direct_answer", False)

        valid_cats = {"stats", "clean", "transform", "viz", "feature", "nlp", "analysis"}
        categories = [c for c in categories if c in valid_cats][:3]

        log.info("  router → categories=%s, direct_answer=%s", categories, direct)
        return categories, bool(direct)
    except Exception as e:
        log.warning("  router failed (%s), falling back to full catalog", e)
        return [], False


def _build_focused_catalog(categories: list[str]) -> str:
    """Build a catalog containing only the selected sub-categories."""
    sections = _get_catalog_sections()
    if not categories:
        return HANDLER_CATALOG  # fallback to full catalog

    parts = [sections[cat] for cat in categories if cat in sections]
    return "\n\n".join(parts)


# ── Stage 2: Focused planner ───────────────────────────────────────────────

def _format_history(history: list | None) -> str:
    """Format recent conversation history for the planner prompt."""
    if not history:
        return "(no previous messages)"
    lines: list[str] = []
    for msg in history[-6:]:
        role = msg.role if hasattr(msg, "role") else msg.get("role", "?")
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        # Truncate long messages
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no previous messages)"


def plan_steps(
    user_message: str,
    df_context: str,
    llm,
    model_id: str | None = None,
    history: list | None = None,
) -> dict:
    """Two-stage planning: route → focused plan.

    Stage 1: Lightweight router (max_tokens=200) classifies into 1-3 sub-categories
    Stage 2: Planner sees only relevant handlers (~25-35 instead of 56-64)
             + conversation history for follow-up context
    """
    # Extract column names for router
    col_match = _re.findall(r"[|]?\s*(\w+)\s*:", df_context[:500])
    columns = col_match[:15] if col_match else []
    has_data = bool(columns) and bool(df_context.strip())

    # Stage 1: Route (lightweight, ~200 tokens)
    categories, is_direct = _route_categories(user_message, columns, model_id, history)

    # Stage 1.5: If router says direct_answer but data is loaded,
    # verify with a second AI call — is the question really unrelated?
    if is_direct and has_data:
        is_about_data = _verify_direct_answer(user_message, columns, model_id)
        if is_about_data:
            log.info("  router said direct_answer but AI verification says it's about data → overriding")
            categories = ["stats", "analysis"]
            is_direct = False

    if is_direct:
        log.info("  router → direct_answer (verified: not about data)")
        return {
            "understanding": "Not about the dataset",
            "output_type": "text",
            "direct_answer": True,
            "steps": [],
        }

    # If router returned empty categories (parse failure), use broad fallback
    if not categories:
        log.info("  router returned empty categories → broad fallback")
        categories = ["stats", "analysis"]

    # Stage 2: Build focused catalog and plan
    focused_catalog = _build_focused_catalog(categories)
    handler_count = focused_catalog.count("| ")
    log.info("  focused catalog: %d categories, ~%d handler rows", len(categories), handler_count // 3)

    prompt = PLANNER_PROMPT.format(
        user_message=user_message,
        df_context=df_context,
        handler_catalog=focused_catalog,
        conversation_history=_format_history(history),
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

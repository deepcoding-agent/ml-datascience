# Handler Feature Reference -- 350 Pre-built Handlers

Complete reference for all pre-built handlers in the PrepPilot DS-Agent.
7 categories x 50 handlers each. The AI planner selects handlers by `category.id`.

The two-stage routing system classifies each user message into 1-3 categories first,
then the focused planner only sees handlers from those categories (~50-150 instead of 350).

---

## Stats (50 handlers)

Instant read-only queries -- no dataset modification.

| ID | Description | Params |
|----|-------------|--------|
| `stats.anova_test` | One-way ANOVA: compare numeric column across multiple groups | `column?`, `group_column?` |
| `stats.cardinality_report` | Unique ratio analysis -- labels each column as ID-like/Binary/High/Low/Medium | -- |
| `stats.chi2_test` | Chi-squared independence test between two categorical columns | `columns?` (list of 2) |
| `stats.class_balance` | Target class distribution + bar chart + balance assessment | `column?` |
| `stats.cluster_tendency` | Hopkins statistic to assess whether data has meaningful clusters | -- |
| `stats.coefficient_variation` | Coefficient of variation (CV = std/mean) per numeric column | -- |
| `stats.column_compare` | Compare two numeric columns statistically (mean, median, std, correlation) | `columns?` (list of 2) |
| `stats.correlation` | Pearson correlation matrix + heatmap chart | -- |
| `stats.correlation_rank` | Spearman rank correlation matrix + heatmap | -- |
| `stats.cross_tab` | Contingency table between 2 columns + heatmap chart | `columns` (list of 2), `normalize?` |
| `stats.data_sample` | Random sample of rows for quick inspection | `n?` (default 5) |
| `stats.describe` | Full column profile: dtype, nulls, mean/std/min/quartiles/max or top/freq | `column?` |
| `stats.distribution_fit` | Fit best distribution (normal/lognormal/exponential) to a column | `column?` |
| `stats.dtypes` | Column data types with null% and unique counts | -- |
| `stats.duplicate_report` | Duplicate row count + sample of duplicates | -- |
| `stats.entropy_report` | Shannon entropy per column (higher = more diverse) | -- |
| `stats.frequency_table` | Frequency table with cumulative percentage | `column?` |
| `stats.gini_report` | Gini impurity per categorical column (0 = pure, 1 = max diversity) | -- |
| `stats.group_stats` | Descriptive stats per group (groupby + describe) | `column?`, `value_column?` |
| `stats.iqr_report` | Interquartile range (Q1, Q3, IQR) per numeric column | -- |
| `stats.ks_test` | Kolmogorov-Smirnov normality test per numeric column | `column?` |
| `stats.kurtosis` | Kurtosis per numeric column with shape label (heavy/light-tailed) | -- |
| `stats.mann_whitney` | Mann-Whitney U test (non-parametric alternative to t-test) | `column?`, `group_column?` |
| `stats.memory_report` | Memory usage breakdown per column and total | -- |
| `stats.missing_pattern` | Analyze which columns tend to be missing together (co-occurrence) | -- |
| `stats.mode_report` | Mode values per column with mode count and percentage | -- |
| `stats.mutual_info_report` | Mutual information scores between all numeric features and a target | `column?` (target) |
| `stats.normality_comprehensive` | Multiple normality tests (Shapiro-Wilk, K-S, D'Agostino) combined | `column?` |
| `stats.normality_test` | Shapiro-Wilk normality test per numeric column with skewness + kurtosis | `column?` |
| `stats.null_report` | Null count + percentage per column, sorted by severity | -- |
| `stats.outlier_report` | IQR-based outlier count per column with bounds | -- |
| `stats.pairwise_stats` | Pairwise statistics (correlation, covariance) for all numeric pairs | -- |
| `stats.percentile` | Custom percentile report (p1, p5, p10, p25, p50, p75, p90, p95, p99) | `column?`, `quantiles?` |
| `stats.quantile_detail` | Detailed quantile report: 1, 5, 10, 25, 50, 75, 90, 95, 99 percentiles | `column?` |
| `stats.range_report` | Range (max - min) per numeric column | -- |
| `stats.ratio_report` | Compute mean ratios between numeric column pairs | -- |
| `stats.shape` | Row/column count, memory, numeric/categorical breakdown, nulls, duplicates | -- |
| `stats.skewness` | Skewness per numeric column sorted by magnitude | -- |
| `stats.sparsity_report` | Sparsity analysis: fraction of zero/null values per column | -- |
| `stats.stability_report` | Feature stability: split data in half, compare first-half vs second-half stats | -- |
| `stats.summary_extended` | Extended summary: mean, median, mode, std, var, range, IQR, skew, kurtosis | -- |
| `stats.t_test` | Independent t-test: compare numeric column across two groups | `column?`, `group_column?` |
| `stats.time_stats` | Time-based statistics for datetime columns (range, gaps, frequency) | `column?` |
| `stats.top_bottom_values` | Show top N and bottom N values of a column side by side | `column?`, `n?` |
| `stats.top_correlations` | Top N most correlated feature pairs (easier to read than matrix) | `n?` (default 10) |
| `stats.unique_values` | Unique count per column sorted descending | -- |
| `stats.value_counts` | Top N value frequencies for a column | `column?`, `n?` (default 10) |
| `stats.variance_report` | Variance per numeric column sorted descending | -- |
| `stats.z_score_report` | Z-score analysis: flag extreme z-scores per column | `column?` |
| `stats.zero_report` | Count of zero/empty values per column | -- |

---

## Clean (50 handlers)

Data cleaning operations -- most produce `output_type: "generate"`.

| ID | Description | Params |
|----|-------------|--------|
| `clean.cap_outliers_percentile` | Cap outliers using percentile bounds (e.g., 1st and 99th) | `column?`, `lower?`, `upper?` |
| `clean.change_dtype` | Cast a column to: int, float, str, bool, datetime, category | `column`, `dtype` |
| `clean.clean_column_names` | Remove special chars, spaces to underscore, lowercase all column names | -- |
| `clean.clean_currency` | Parse currency strings ($1,234.56) into numeric values | `column?` |
| `clean.clean_phone_numbers` | Normalize phone number formats to digits only | `column?` |
| `clean.clean_text_whitespace` | Normalize all whitespace in text columns (collapse multiple spaces) | `column?` |
| `clean.clip_outliers` | Cap outlier values using IQR or z-score (values are clipped, not removed) | `column?`, `method?` |
| `clean.dedup_keep_latest` | Remove duplicates keeping the last occurrence (by timestamp or index) | `column?` |
| `clean.deduplicate_by` | Remove duplicates by specific column(s), keep first or last | `column?`, `columns?`, `keep?` |
| `clean.drop_column` | Drop one or more columns | `column`, `columns?` |
| `clean.drop_constant` | Drop columns where all values are identical (zero information) | -- |
| `clean.drop_id_columns` | Auto-detect and drop ID-like columns (sequential, high unique ratio) | -- |
| `clean.drop_nulls` | Drop rows with nulls; optionally drop high-null columns first (threshold) | `column?`, `threshold?` |
| `clean.fill_forward_backward` | Fill nulls using forward fill then backward fill combined | `column?` |
| `clean.fill_interpolate` | Fill nulls via interpolation: linear, ffill, or bfill | `column?`, `method?` |
| `clean.fill_median_by_group` | Fill nulls with group-specific median (more accurate than global) | `column?`, `group_column?` |
| `clean.fill_mode` | Fill nulls with the mode (most frequent value) | `column?` |
| `clean.fill_nulls` | Fill nulls with strategy: auto (skew-aware), median, mean, mode, or zero | `column?`, `strategy?` |
| `clean.fill_with_distribution` | Fill nulls by sampling from the column's distribution | `column?` |
| `clean.fill_with_value` | Fill nulls with a specific constant (e.g., -1, "Unknown", "N/A") | `column?`, `value` |
| `clean.fix_boolean` | Convert boolean-like columns (yes/no, true/false, 0/1) to proper bool | `column?` |
| `clean.fix_date_outliers` | Fix impossible dates (e.g., year 9999, future dates) | `column?` |
| `clean.fix_dtypes` | Auto-detect and convert string columns to numeric or datetime | -- |
| `clean.fix_encoding` | Fix character encoding issues (mojibake) in text columns | `column?` |
| `clean.fix_mixed_types` | Resolve columns with mixed data types to the dominant type | `column?` |
| `clean.fix_numeric_strings` | Convert formatted numeric strings ('$1,234' / '1.234,56') to float | `column?` |
| `clean.fix_whitespace_names` | Fix column names with leading/trailing whitespace | -- |
| `clean.lowercase_columns` | Normalize all column names to snake_case | -- |
| `clean.lowercase_values` | Lowercase all string values in column(s) | `column?` |
| `clean.map_values` | Recode/map values using a dict (e.g., {"M": "Male", "F": "Female"}) | `column`, `mapping` |
| `clean.normalize_text_case` | Normalize text to title case, upper case, or lower case | `column?`, `case?` |
| `clean.remove_duplicates` | Remove all duplicate rows | -- |
| `clean.remove_emails` | Remove email addresses from text columns | `column?` |
| `clean.remove_empty_rows` | Remove rows where all values are null or empty string | -- |
| `clean.remove_high_null_cols` | Drop columns exceeding a null percentage threshold | `threshold?` (default 0.5) |
| `clean.remove_html_tags` | Strip HTML tags from text columns | `column?` |
| `clean.remove_negative` | Remove rows with negative values in numeric columns | `column?` |
| `clean.remove_non_ascii` | Remove non-ASCII characters from text columns | `column?` |
| `clean.remove_outliers` | Remove rows containing outlier values (IQR or z-score) | `column?`, `method?` |
| `clean.remove_rare_categories` | Merge rare categories (below threshold) into "Other" | `column?`, `threshold?` |
| `clean.remove_special_chars` | Remove special characters from text columns | `column?` |
| `clean.remove_urls` | Remove URLs from text columns | `column?` |
| `clean.remove_zero_rows` | Remove rows where a specific column is zero | `column?` |
| `clean.rename_column` | Rename a column | `column` (old), `new_name` |
| `clean.replace_values` | Replace specific values (e.g., "?" -> NaN) in one or all columns | `column?`, `old_value`, `new_value` |
| `clean.reset_index` | Reset DataFrame index to 0-based sequential | -- |
| `clean.split_name` | Split a full name column into first and last name columns | `column?` |
| `clean.standardize_categories` | Standardize category labels (fix typos, merge similar) | `column?`, `mapping?` |
| `clean.standardize_dates` | Convert date columns to a consistent format | `column?`, `format?` |
| `clean.strip_whitespace` | Trim leading/trailing whitespace from all string columns | -- |

---

## Transform (50 handlers)

Data transformation and reshaping -- most produce `output_type: "generate"`.

| ID | Description | Params |
|----|-------------|--------|
| `transform.add_column` | Add new column via pandas eval expression | `column` (name), `expression` |
| `transform.apply_expr` | Apply a pandas expression to the entire DataFrame | `expression` |
| `transform.assign_value` | Set all values in a column to a constant | `column`, `value` |
| `transform.bin_column` | Equal-width binning into N bins | `column`, `n?` (default 5) |
| `transform.clip` | Clip column values to min/max bounds | `column`, `lower?`, `upper?` |
| `transform.concat_columns` | Concatenate multiple columns into one new column | `columns`, `separator?`, `new_name?` |
| `transform.cross_join` | Cross join (cartesian product) with a grouped summary | `column?` |
| `transform.cumulative` | Cumulative sum/count/max/min | `column?`, `agg?` |
| `transform.drop_rows` | Drop rows by index range or specific indices | `start?`, `end?`, `indices?` |
| `transform.duplicate_column` | Create a copy of an existing column with a new name | `column`, `new_name?` |
| `transform.encode_binary` | Binary encode: 0/1 for two-class columns | `column?` |
| `transform.encode_label` | Label-encode categorical columns (A->0, B->1, C->2) | `column?` |
| `transform.encode_onehot` | One-hot encode categorical columns (drop_first=True) | `column?` |
| `transform.encode_ordinal` | Ordinal encoding with specified category order | `column`, `order?` |
| `transform.explode` | Explode a list/array column into separate rows | `column` |
| `transform.fill_forward` | Forward fill (ffill) null values | `column?` |
| `transform.filter` | Filter rows by condition (==, !=, >, <, >=, <=) | `column`, `operator`, `value` |
| `transform.flatten_columns` | Flatten multi-level column index to single level | -- |
| `transform.groupby_agg` | Group by column + aggregate (count/sum/mean/max/min) | `column`, `agg?` |
| `transform.head` | First N rows | `n?` (default 10) |
| `transform.inject_null` | Inject random NaN values (for testing) | `value` (percentage) |
| `transform.interpolate_values` | Interpolate numeric values using various methods | `column?`, `method?` |
| `transform.melt` | Unpivot wide -> long format | `id_vars?`, `value_vars?` |
| `transform.merge` | Merge/join by key column (self-join with aggregation) | `column`, `how?` |
| `transform.nlargest` | Top N rows by column value | `column?`, `n?` |
| `transform.normalize_pct` | Normalize numeric columns to percentage of total | `column?` |
| `transform.nsmallest` | Bottom N rows by column value | `column?`, `n?` |
| `transform.pct_change` | Percentage change between consecutive rows | `column?` |
| `transform.pivot` | Pivot table (aggregate + reshape) | `index`/`column`, `columns?`, `values?`, `agg?` |
| `transform.qcut` | Quantile-based binning (equal-frequency bins) | `column`, `n?` (default 4) |
| `transform.rank` | Add rank column based on numeric values | `column?`, `ascending?` |
| `transform.reorder_columns` | Reorder columns in specified order | `columns` (list) |
| `transform.resample` | Resample time series data to a different frequency | `column?`, `freq?`, `agg?` |
| `transform.rolling` | Rolling/moving window: mean, sum, std | `column?`, `window?`, `agg?` |
| `transform.round_values` | Round numeric columns to N decimal places | `column?`, `decimals?` |
| `transform.sample_rows` | Random sample of N rows | `n?` (default 10) |
| `transform.scale_minmax` | MinMaxScaler: normalize to [0, 1] | -- |
| `transform.scale_robust` | RobustScaler: median-centered, IQR-scaled (resistant to outliers) | -- |
| `transform.scale_standard` | StandardScaler: z-score normalization (mean=0, std=1) | -- |
| `transform.shift_column` | Shift column values up or down by N rows | `column`, `periods?` |
| `transform.shuffle` | Randomly shuffle all rows | `seed?` |
| `transform.sort` | Sort by column ascending or descending | `column`, `ascending?` |
| `transform.split_column` | Split a string column by delimiter into multiple new columns | `column`, `delimiter?` |
| `transform.stack_columns` | Stack multiple columns into a single column (name/value pairs) | `columns?` |
| `transform.tail` | Last N rows | `n?` (default 10) |
| `transform.train_test_split` | Split dataset into train/test sets with _split column | `test_size?`, `seed?`, `column?` (stratify) |
| `transform.transpose` | Transpose the DataFrame (swap rows and columns) | -- |
| `transform.unstack_column` | Unstack a column from long to wide format | `column?` |
| `transform.where` | Conditional assignment: set values based on a condition | `column`, `condition`, `value` |
| `transform.winsorize` | Winsorize: clip extreme values to percentile bounds | `column?`, `limits?` |

---

## Visualization (50 handlers)

All charts use Plotly with a unified minimal theme (plotly_white, `#FB8C3C` accent, Inter font).

| ID | Description | Params |
|----|-------------|--------|
| `viz.area_chart` | Area chart for trends | `column?` |
| `viz.bar_chart` | Bar chart of value counts (top 15) | `column?`, `percentage?` |
| `viz.box_plot` | Box plot (single column or multi-column comparison) | `column?` |
| `viz.bubble_chart` | Bubble chart using 3 numeric columns (x, y, size) | -- |
| `viz.candlestick` | Candlestick chart for OHLC financial data | `columns?` |
| `viz.comparison_bar` | Side-by-side bar chart comparing two columns or groups | `columns?` |
| `viz.contour_plot` | 2D contour/density plot for two numeric columns | `columns?` |
| `viz.correlation_scatter` | Scatter plot with regression line and correlation value | `columns?` |
| `viz.count_plot` | Count/frequency bar chart (top 20) | `column?` |
| `viz.cumulative_line` | Cumulative distribution line chart | `column?` |
| `viz.density_plot` | KDE density plot (optionally grouped) | `column?`, `group?` |
| `viz.distribution` | Histogram + marginal box plot | `column?` |
| `viz.donut_chart` | Donut chart (pie with hole in center) | `column?` |
| `viz.dot_plot` | Dot plot (Cleveland dot chart) for comparing values | `column?` |
| `viz.dual_axis` | Dual Y-axis chart for two numeric columns with different scales | `columns?` |
| `viz.ecdf_plot` | Empirical cumulative distribution function plot | `column?` |
| `viz.error_bar_chart` | Bar chart with error bars (mean +/- std per group) | `column?`, `group?` |
| `viz.funnel_chart` | Funnel chart for stages/sequential data | `column?`, `value_column?` |
| `viz.gauge_chart` | Gauge/speedometer chart for a single metric | `column?`, `value?` |
| `viz.grouped_bar` | Grouped bar chart comparing categories across groups | `columns?` |
| `viz.heatmap` | Correlation heatmap (RdYlBu color scale) | -- |
| `viz.histogram` | Histogram with marginal box plot | `column?` |
| `viz.histogram_2d` | 2D histogram (heatmap-style binned scatter) | `columns?` |
| `viz.line_chart` | Line chart | `column?` |
| `viz.lollipop_chart` | Lollipop chart (dot on stick) for ranked data | `column?` |
| `viz.marimekko` | Marimekko/mosaic chart for two categorical variables | `columns?` |
| `viz.missing_heatmap` | Missing values pattern heatmap | -- |
| `viz.null_bar` | Bar chart showing null counts per column | -- |
| `viz.pairplot` | Scatter matrix of numeric columns (max 5) | -- |
| `viz.parallel_coords` | Parallel coordinates plot (max 6 numeric cols) | -- |
| `viz.pareto_chart` | Pareto chart (bar + cumulative line, 80/20 rule) | `column?` |
| `viz.percent_bar` | 100% stacked bar chart showing proportions | `columns?` |
| `viz.pie_chart` | Donut pie chart (auto-groups >6 categories into "Other") | `column?` |
| `viz.polar_chart` | Polar/radar area chart | `columns?` |
| `viz.qq_plot` | QQ plot for normality check (data vs normal line) | `column?` |
| `viz.radar_chart` | Radar/spider chart for multi-dimensional comparison | `columns?` |
| `viz.range_plot` | Range plot showing min-max span per category | `column?`, `group?` |
| `viz.ridgeline` | Ridgeline plot (stacked density curves per group) | `column?`, `group?` |
| `viz.sankey_chart` | Sankey flow diagram between two categorical columns | `columns?` |
| `viz.scatter` | Scatter plot (optional color dimension) | `columns?` (list: [x, y, color?]) |
| `viz.stacked_bar` | Stacked bar chart -- group by one column, color by another | `columns?` (list: [x, color]) |
| `viz.step_chart` | Step/staircase chart for discrete changes | `column?` |
| `viz.strip_plot` | Jitter/strip plot showing individual data points | `column?`, `group?` |
| `viz.sunburst` | Sunburst chart for 2 categorical columns | -- |
| `viz.swarm_plot` | Swarm/beeswarm plot (non-overlapping strip) | `column?`, `group?` |
| `viz.time_series` | Time series line chart (requires datetime column) | `column?` |
| `viz.top_n_bar` | Bar chart of top N values for a column | `column?`, `n?` |
| `viz.treemap` | Treemap of categorical column value counts | `column?` |
| `viz.violin_plot` | Violin plot with embedded box | `column?` |
| `viz.waterfall_chart` | Waterfall chart showing cumulative effect of sequential values | `column?` |

---

## Feature Engineering (50 handlers)

Feature creation, selection, and transformation -- most produce `output_type: "generate"`.

| ID | Description | Params |
|----|-------------|--------|
| `feature.abs_transform` | Absolute value transform for numeric columns | `column?` |
| `feature.aggregation_features` | Create group-by aggregation features (mean/std/count per group) | `column?` (group-by), `agg_column?` |
| `feature.auto_feature_select` | Automatic feature selection using multiple methods combined | `column?` (target), `k?` |
| `feature.bin_numeric` | Bin numeric column into categorical bins with labels | `column`, `n?`, `labels?` |
| `feature.boxcox_transform` | Box-Cox power transform (requires positive values) | `column?` |
| `feature.clip_features` | Clip feature values to specified percentile bounds | `column?`, `lower?`, `upper?` |
| `feature.correlation_filter` | Drop features with correlation above threshold | `value?` (default 0.95) |
| `feature.count_encode` | Encode categories by their occurrence count | `column?` |
| `feature.cyclical_encode` | sin/cos encoding for cyclical features (month, dayofweek, hour) | `column`, `period?` |
| `feature.datetime_features` | Extract year, month, day, dayofweek, hour from datetime columns | `column?` |
| `feature.diff_features` | Create difference features (first/second order) for time series | `column?`, `periods?` (default 1) |
| `feature.distance_from_mean` | Compute absolute distance from mean for each value | `column?` |
| `feature.ewm_features` | Exponentially weighted moving average/std features | `column?`, `span?` |
| `feature.exponential_transform` | Exponential transform (e^x) for numeric columns | `column?` |
| `feature.feature_cross` | Create feature crosses (A_B = concat of two categoricals) | `columns?` |
| `feature.feature_importance` | Random Forest feature importance ranking + horizontal bar chart | `column?` (target) |
| `feature.frequency_encode` | Encode categorical values by their frequency (normalized) | `column?` |
| `feature.hash_encode` | Hash-based encoding for very high-cardinality categoricals | `column?`, `n_components?` |
| `feature.interaction_features` | Create interaction terms between specified columns | `columns?` |
| `feature.is_holiday` | Create binary feature for holiday dates | `column?` |
| `feature.is_null_features` | Create binary indicators for null values per column | `column?` |
| `feature.is_weekend` | Create binary feature for weekend dates | `column?` |
| `feature.is_zero_features` | Create binary indicators for zero values per column | `column?` |
| `feature.kbins_discretize` | KBins discretization (uniform, quantile, or kmeans strategy) | `column?`, `n?`, `strategy?` |
| `feature.label_binarize` | Binarize labels for multi-class into one-vs-all format | `column?` |
| `feature.lag_features` | Create lag/shift features for time series data | `column?`, `lags?` (list or int) |
| `feature.log1p_transform` | log1p transform (handles zeros better than log) | `column?` |
| `feature.log_transform` | log1p transform for highly skewed columns (skew > 1) | `column?` |
| `feature.mutual_info` | Mutual information scores (works for numeric + categorical) + bar chart | `column?` (target) |
| `feature.ordinal_encode` | Ordinal encoding with custom category order | `column?`, `order?` |
| `feature.pca` | PCA + scatter plot with explained variance | `n?` (components, default 2) |
| `feature.polynomial_features` | Create interaction features (col_a x col_b) for numeric columns | `columns?` |
| `feature.power_transform` | Yeo-Johnson or Box-Cox power transform (standardized) | `column?`, `method?` |
| `feature.quantile_transform` | Quantile transform to uniform or normal distribution | `column?`, `distribution?` |
| `feature.rank_transform` | Replace values with their rank (ordinal ranking) | `column?` |
| `feature.rare_category_encode` | Encode rare categories (below threshold) as a single group | `column?`, `threshold?` |
| `feature.ratio_features` | Create ratio features between numeric column pairs (a / b) | `columns?` |
| `feature.reciprocal_transform` | Reciprocal transform (1/x) for numeric columns | `column?` |
| `feature.rolling_stats_features` | Create rolling statistics features (mean, std, min, max) | `column?`, `window?` |
| `feature.select_k_best` | Select top K features using f_classif/f_regression + chart | `column?` (target), `k?` |
| `feature.sin_cos_hour` | sin/cos encoding specifically for hour-of-day | `column?` |
| `feature.sqrt_transform` | Square root transform for moderately skewed columns | `column?` |
| `feature.target_binary_encode` | Binary target encoding (leave-one-out) | `column?` (target), `encode_column?` |
| `feature.target_encode` | Mean/target encoding for high-cardinality categoricals | `column?` (target), `encode_column?` |
| `feature.text_features` | Extract text statistics: length, word count, digit count, uppercase ratio | `column?` |
| `feature.time_since` | Compute time elapsed since a reference date | `column?`, `reference?` |
| `feature.variance_filter` | Drop features with variance below threshold | `value?` (default 0.01) |
| `feature.winsorize` | Winsorize features to reduce extreme value influence | `column?`, `limits?` |
| `feature.yeo_johnson_transform` | Yeo-Johnson power transform (handles negative values) | `column?` |
| `feature.zscore_features` | Add z-score columns for numeric features | `column?` |

---

## NLP / Text Preprocessing (50 handlers)

Text cleaning, tokenization, vectorization, and NLP feature engineering.
Most produce `output_type: "generate"`.

| ID | Description | Params |
|----|-------------|--------|
| `nlp.bow` | Bag-of-words vectorization (count matrix) | `column?`, `max_features?` |
| `nlp.char_features` | Character-level features: char count, special chars, digits, uppercase ratio | `column?` |
| `nlp.class_balance_text` | Analyze class balance for text classification datasets | `column?` (label column) |
| `nlp.collocations` | Extract frequent word collocations (bigrams/trigrams) | `column?`, `n?` |
| `nlp.doc_term_matrix` | Create document-term matrix (sparse representation) | `column?`, `max_features?` |
| `nlp.emoji_features` | Extract emoji-related features (count, types, has_emoji) | `column?` |
| `nlp.hash_vectorize` | Hash-based vectorization (fixed-size, memory-efficient) | `column?`, `n_features?` |
| `nlp.keyword_extract` | Extract top keywords using TF-IDF scoring | `column?`, `n?` |
| `nlp.language_detect` | Detect language of text using Unicode character ranges | `column?` |
| `nlp.ngrams` | Generate n-grams (bigrams, trigrams) from text | `column?`, `n?` (default 2) |
| `nlp.readability_score` | Compute text readability metrics (avg word length, sentence length) | `column?` |
| `nlp.regex_extract` | Extract patterns (emails, URLs, hashtags, numbers) via regex | `column?`, `pattern?` |
| `nlp.remove_stopwords` | Remove English stopwords from text columns | `column?`, `extra_words?` |
| `nlp.sentence_features` | Sentence-level features: sentence count, avg sentence length | `column?` |
| `nlp.sentiment_score` | Lexicon-based sentiment analysis (positive/negative/neutral) | `column?` |
| `nlp.spelling_features` | Spelling-related features: unusual word ratio, avg word length | `column?` |
| `nlp.text_augment` | Augment text data by synonym replacement or random insertion | `column?`, `method?` |
| `nlp.text_chunk` | Split text into fixed-size chunks | `column?`, `chunk_size?` |
| `nlp.text_clean` | Clean text: lowercase, remove HTML/URLs/emails/punctuation/numbers | `column?`, `strategy?` |
| `nlp.text_concat` | Concatenate multiple text columns into one | `columns?`, `separator?` |
| `nlp.text_count_pattern` | Count occurrences of a regex pattern in text | `column?`, `pattern?` |
| `nlp.text_dedup` | Fuzzy text deduplication based on similarity threshold | `column?`, `threshold?` |
| `nlp.text_dedup_exact` | Exact text deduplication (remove identical texts) | `column?` |
| `nlp.text_diversity_index` | Compute vocabulary diversity (type-token ratio) per document | `column?` |
| `nlp.text_encode` | Encode text to numeric using label encoding | `column?` |
| `nlp.text_extract_numbers` | Extract all numbers from text into a new column | `column?` |
| `nlp.text_filter` | Filter rows based on text content (contains/not contains) | `column?`, `pattern?`, `exclude?` |
| `nlp.text_label_rules` | Apply rule-based labeling using keyword lists | `column?`, `rules?` |
| `nlp.text_length_dist` | Analyze text length distribution + histogram chart | `column?` |
| `nlp.text_mask_pii` | Mask personally identifiable information (emails, phones, names) | `column?` |
| `nlp.text_ngram_frequency` | Frequency analysis of n-grams with bar chart | `column?`, `n?` |
| `nlp.text_normalize` | Normalize text: Unicode NFKD, accents removal, stemming | `column?` |
| `nlp.text_oversample` | Oversample minority text classes for balanced training | `column?` (label column) |
| `nlp.text_pos_patterns` | Extract part-of-speech-like patterns using regex heuristics | `column?` |
| `nlp.text_remove_rare` | Remove words that appear fewer than N times across corpus | `column?`, `min_count?` |
| `nlp.text_replace` | Find and replace text patterns using regex | `column?`, `pattern?`, `replacement?` |
| `nlp.text_similarity` | Compute pairwise text similarity using character n-grams | `column?` |
| `nlp.text_split_sentences` | Split text into individual sentences (one per row) | `column?` |
| `nlp.text_stratified_sample` | Stratified sampling of text data by label | `column?` (label), `n?` |
| `nlp.text_summary_report` | Comprehensive text column summary: lengths, vocab, patterns | `column?` |
| `nlp.text_to_paragraphs` | Split text into paragraphs based on newlines | `column?` |
| `nlp.text_truncate_pad` | Truncate or pad text to a fixed length | `column?`, `max_length?` |
| `nlp.text_unique_words` | Extract unique words per document | `column?` |
| `nlp.text_window` | Sliding window over text (for context extraction) | `column?`, `window_size?` |
| `nlp.tfidf` | TF-IDF vectorization | `column?`, `max_features?` |
| `nlp.tokenize` | Tokenize text into words (whitespace + punctuation split) | `column?` |
| `nlp.vocab_stats` | Vocabulary statistics: total words, unique words, hapax legomena | `column?` |
| `nlp.word_cloud` | Generate word frequency data suitable for word cloud visualization | `column?` |
| `nlp.word_frequency` | Word frequency table sorted by count + bar chart | `column?`, `n?` |
| `nlp.word_overlap` | Compute word overlap between pairs of text columns | `columns?` |

---

## Analysis (50 handlers)

Smart, high-level analytical handlers that combine multiple operations and return
rich, insight-driven results with charts and formatted summaries.

| ID | Description | Params |
|----|-------------|--------|
| `analysis.ab_test` | A/B test analysis with statistical significance testing | `column?`, `group_column?`, `value_column?` |
| `analysis.anomaly_detect` | Detect anomalies using IQR, z-score, or Isolation Forest | `column?`, `method?` |
| `analysis.auto_eda` | Automated exploratory data analysis with key insights | -- |
| `analysis.benchmark_compare` | Compare dataset metrics against typical benchmarks | -- |
| `analysis.bootstrap_ci` | Bootstrap confidence intervals for numeric columns | `column?`, `n_bootstrap?` |
| `analysis.bottom_n_analysis` | Deep analysis of the bottom N rows by a column | `column?`, `n?` |
| `analysis.categorical_analysis` | Comprehensive analysis of categorical columns (distribution, chi2) | `column?` |
| `analysis.categorical_target_crosstab` | Cross-tabulation of categorical features vs target with insights | `column?` (target) |
| `analysis.change_point_detect` | Detect change points in time series data | `column?` |
| `analysis.cluster_kmeans` | K-Means clustering with auto-K selection + scatter plot | `n?` (clusters) |
| `analysis.cluster_profile` | Profile each cluster: mean, mode, distinguishing features | `column?` (cluster label) |
| `analysis.cohort_analysis` | Cohort analysis based on time periods | `column?` (date), `value_column?` |
| `analysis.compare_columns` | Statistical comparison between two columns with visualization | `columns?` |
| `analysis.compare_extremes` | Compare rows with highest and lowest value of a column | `column?` |
| `analysis.concentration_analysis` | Concentration analysis (Gini coefficient, Lorenz curve) | `column?` |
| `analysis.correlation_insights` | Top correlations with interpretive insights and warnings | -- |
| `analysis.correlation_network` | Correlation network graph (nodes = features, edges = strong correlations) | `threshold?` |
| `analysis.cross_correlation` | Cross-correlation analysis between two time series | `columns?` |
| `analysis.data_completeness` | Data completeness score per column and overall | -- |
| `analysis.data_quality` | Comprehensive data quality report (nulls, types, outliers, duplicates) | -- |
| `analysis.data_readiness_score` | ML readiness score: assess if data is ready for modeling | `column?` (target) |
| `analysis.deep_profile` | Deep statistical profile of a single column with auto-visualization | `column?` |
| `analysis.diminishing_returns` | Identify diminishing returns in feature-target relationships | `column?`, `value_column?` |
| `analysis.distribution_analysis` | Comprehensive distribution analysis with best-fit test | `column?` |
| `analysis.effect_size` | Cohen's d effect size between two groups | `column?`, `group_column?` |
| `analysis.feature_drift` | Detect feature drift by comparing first-half vs second-half distributions | `column?` |
| `analysis.feature_interaction` | Analyze interaction effects between two features on a target | `columns?`, `target?` |
| `analysis.feature_selection_auto` | Automatic feature selection using combined scoring methods | `column?` (target) |
| `analysis.gap_analysis` | Identify gaps in sequential or time-based data | `column?` |
| `analysis.group_insights` | Compare numeric statistics across groups with interesting differences | `column?`, `value_column?` |
| `analysis.hypothesis_test` | Automated hypothesis test selection based on data characteristics | `column?`, `group_column?` |
| `analysis.missing_value_analysis` | In-depth missing value analysis: patterns, correlations, impact | -- |
| `analysis.multicollinearity_check` | VIF-based multicollinearity check for numeric features | -- |
| `analysis.numeric_summary` | Rich numeric summary: distribution shape, outliers, quality assessment | `column?` |
| `analysis.outlier_isolation_forest` | Isolation Forest-based outlier detection with visualization | `column?` |
| `analysis.pareto_analysis` | Pareto analysis (80/20 rule) for categorical values | `column?` |
| `analysis.pca_2d` | PCA-based 2D projection with explained variance chart | -- |
| `analysis.percentile_analysis` | Percentile-based analysis with insights on distribution shape | `column?` |
| `analysis.prediction_baseline` | Establish baseline prediction metrics (mean, median, mode) | `column?` (target) |
| `analysis.regression_quick` | Quick linear regression: R2, coefficients, residual plot | `column?` (target) |
| `analysis.rfm_analysis` | RFM (Recency, Frequency, Monetary) customer segmentation | `columns?` |
| `analysis.sample_bias_check` | Check for sampling bias by comparing subgroup distributions | `column?` |
| `analysis.seasonality_detect` | Detect seasonal patterns in time series data | `column?` |
| `analysis.segment_analysis` | Segment data by a categorical column and compare group profiles | `column?` |
| `analysis.sensitivity_analysis` | Sensitivity analysis: how changes in features affect the target | `column?` (target) |
| `analysis.survival_curve` | Kaplan-Meier-style survival curve analysis | `column?` (time), `event_column?` |
| `analysis.target_analysis` | Comprehensive analysis of a target column: distribution, correlations, predictors | `column?` |
| `analysis.top_n_analysis` | Deep analysis of the top N rows by a column | `column?`, `n?` |
| `analysis.trend_detect` | Detect trends (increasing/decreasing/stable) in numeric columns | `column?` |
| `analysis.variance_analysis` | Variance decomposition: within-group vs between-group variance | `column?`, `group_column?` |

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

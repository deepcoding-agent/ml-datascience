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
| clean.fix_numeric_strings | convert "$1,234" / "1.234,56" to numeric | column? |
| clean.clean_column_names | remove special chars, spaces→underscore, lowercase | (none) |
| clean.remove_empty_rows | remove rows where all values are null/empty | (none) |
| clean.fill_mode | fill nulls with mode (most frequent value) | column? |
| clean.fill_forward_backward | ffill then bfill to fill nulls | column? |
| clean.fix_boolean | standardize yes/no/true/false/Y/N/1/0 → bool | column? |
| clean.fix_encoding | fix mojibake/encoding issues (special chars) | column? |
| clean.remove_html_tags | strip HTML/XML tags from strings | column? |
| clean.clean_currency | clean currency strings ($, €, ¥, commas) → float | column? |
| clean.standardize_dates | parse mixed date formats to consistent format | column?, format? (default %Y-%m-%d) |
| clean.remove_non_ascii | remove non-ASCII characters | column? |
| clean.remove_special_chars | remove special characters (keep alphanumeric + spaces) | column?, keep? (regex pattern) |
| clean.normalize_text_case | normalize to title/upper/lower/sentence case | column?, case? (lower/upper/title/sentence) |
| clean.cap_outliers_percentile | cap at Nth percentile | column?, lower? (default 1), upper? (default 99) |
| clean.fill_median_by_group | fill nulls with group-level median | column (group col), value_column? |
| clean.remove_zero_rows | remove rows where column(s) are zero | column? |
| clean.remove_negative | remove rows with negative values | column? |
| clean.standardize_categories | merge similar categories (strip, lower, map) | column, mapping? (dict) |
| clean.remove_high_null_cols | drop columns above null threshold | threshold? (default 0.5) |
| clean.clean_phone_numbers | standardize phone numbers to digits-only | column |
| clean.split_name | split "John Doe" → first_name, last_name | column |
| clean.fix_whitespace_names | fix " John  Doe " → "John Doe" | column? |
| clean.remove_urls | remove URLs from text | column? |
| clean.remove_emails | remove email addresses from text | column? |
| clean.fix_mixed_types | convert mixed-type columns to consistent type | column? |
| clean.fill_with_distribution | fill nulls by sampling from column distribution | column?, seed? |
| clean.remove_rare_categories | replace categories with < N occurrences with "Other" | column, min_count? (default 5), replacement? |
| clean.dedup_keep_latest | dedup by column keeping latest by sort column | column (key), date_column (sort) |
| clean.fix_date_outliers | remove/clip dates outside valid range | column, min_date?, max_date?, action? (remove/clip) |
| clean.clean_text_whitespace | normalize all whitespace (double spaces, tabs, newlines) | column? |

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
| transform.merge | merge/group by key column, aggregate numeric cols | column (key), how? (inner/left/right/outer) |
| transform.transpose | transpose DataFrame (swap rows and columns) | (none) |
| transform.drop_rows | drop rows by index range or specific indices | start?, end?, indices? (list) |
| transform.shuffle | randomly shuffle all rows | seed? (default 42) |
| transform.train_test_split | split into train/test sets, adds _split column | test_size? (default 0.2), seed?, column? (stratify) |
| transform.clip | clip numeric values to min/max bounds | column?, min?, max? |
| transform.where | replace values where condition is NOT met | column, operator, value, replacement? |
| transform.explode | explode comma-separated/list column into separate rows | column, delimiter? |
| transform.encode_binary | encode column as 0/1 based on threshold or specific value | column, threshold?, value? |
| transform.pct_change | compute percentage change between consecutive rows | column?, periods? (default 1) |
| transform.normalize_pct | normalize numeric columns to percentages | axis? (columns=row-wise, index=col-wise) |
| transform.apply_expr | apply math expression to create new column (e.g. "price / area") | expression, new_name? |
| transform.flatten_columns | flatten multi-level columns to snake_case | (none) |
| transform.resample | resample time series to different frequency (D/W/M/Q/Y) | column? (date col), freq? (default M), agg? (mean/sum) |
| transform.cross_join | cartesian product of unique values from two columns | columns (list of 2) |

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

### NLP / Text Preprocessing (for text/language datasets → output_type "generate" unless noted)
| id | what it does | params |
|----|-------------|--------|
| nlp.text_clean | clean text: lowercase, remove HTML/URLs/emails/punctuation/numbers, normalize whitespace | column?, strategy? (all/lowercase/no_punct/no_numbers/no_html/no_urls/no_emails) |
| nlp.remove_stopwords | remove English stopwords from text columns | column?, extra_words? (list) |
| nlp.tokenize | split text into tokens, create token_count column | column? |
| nlp.tfidf | TF-IDF vectorization → N feature columns | column?, n? (max features, default 50) |
| nlp.bow | Bag of Words count vectorization → N feature columns | column?, n? (max features, default 50) |
| nlp.ngrams | extract word n-gram features via TF-IDF | column?, n? (gram size, default 2), max_features? |
| nlp.regex_extract | extract patterns: email/url/hashtag/mention/phone/number/custom | column?, pattern? (email/url/hashtag/mention/phone/number/all), regex? |
| nlp.sentiment_score | lexicon-based sentiment scoring (positive/negative/compound) + chart | column? |
| nlp.word_frequency | top-N word frequency analysis with bar chart (output_type=query) | column?, n? (default 20), remove_stopwords? |
| nlp.text_similarity | cosine similarity matrix using TF-IDF + heatmap (output_type=query) | column? |
| nlp.vocab_stats | vocabulary statistics: unique tokens, TTR, avg word length, hapax (output_type=query) | column? |
| nlp.text_normalize | normalize: strip accents + basic stemming + lowercase | column?, stem? (default true) |
| nlp.language_detect | detect language per row from Unicode character ranges + pie chart | column? |
| nlp.hash_vectorize | feature hashing — fast, memory-efficient text vectorization | column?, n? (features, default 32) |
| nlp.text_encode | encode text as integer sequences (word→ID) for deep learning | column?, max_vocab? (default 5000), max_len? (default 100) |
| nlp.keyword_extract | extract top-N keywords per document using TF-IDF scores | column?, n? (keywords per doc, default 5) |
| nlp.char_features | character-level features: punct/digit/upper/lower/space ratios, avg word length | column? |
| nlp.sentence_features | sentence-level stats: count, avg/min/max length, question/exclamation counts | column? |
| nlp.readability_score | readability metrics: Flesch Reading Ease, Coleman-Liau, ARI | column? |
| nlp.text_dedup | find/remove near-duplicate texts using TF-IDF cosine similarity | column?, threshold? (default 0.9), action? (flag/remove) |
| nlp.emoji_features | extract emoji + emoticon count, ratio, and list per row | column? |
| nlp.text_mask_pii | mask PII: emails, phones, credit cards, SSN, IPs, URLs → [EMAIL] etc. | column? |
| nlp.text_augment | text augmentation: random word delete/swap to expand dataset | column?, n? (copies, default 1), strategy? (delete/swap/mixed) |
| nlp.collocations | find significant word pair collocations ranked by PMI + chart (output_type=query) | column?, n? (default 20) |
| nlp.word_cloud | word frequency treemap visualization (output_type=query) | column?, n? (words, default 40) |
| nlp.text_filter | filter rows by text criteria: min/max length, contains, min words | column?, min_len?, max_len?, min_words?, contains?, not_contains? |
| nlp.class_balance_text | analyze text label class balance + distribution chart (output_type=query) | column (label col), text_column? |
| nlp.text_chunk | split long texts into fixed-size word chunks (new rows per chunk) | column?, chunk_size? (default 200), overlap? (default 20) |
| nlp.spelling_features | OOV/spelling quality features: rare-word count and ratio per row | column? |
| nlp.text_concat | combine multiple text columns into one corpus column | columns? (list), separator? (default " "), new_name? (default "text_combined") |
| nlp.text_replace | find and replace text patterns (regex or literal) | column?, pattern, replacement, mapping? (dict), regex? (default true) |
| nlp.text_split_sentences | split text into individual sentences (new rows per sentence) | column? |
| nlp.text_oversample | oversample minority text classes to balance dataset | column? (label col) |
| nlp.doc_term_matrix | build document-term frequency matrix (full vocab) | column?, n? (max features, default 100) |
| nlp.text_window | extract sliding window contexts around a keyword | column?, keyword, window? (default 5) |
| nlp.text_label_rules | create labels from keyword rules: {label: [keywords]} | column?, mapping (dict), default? |
| nlp.word_overlap | Jaccard word overlap between two columns or consecutive rows | columns? (list of 2), column? |
| nlp.text_truncate_pad | truncate or pad text to fixed word count | column?, max_words? (default 128), pad_token? |
| nlp.text_length_dist | analyze text length distribution with histogram (output_type=query) | column? |
| nlp.text_unique_words | extract words unique to each document (corpus-level rarity) | column? |
| nlp.text_dedup_exact | fast exact-match text deduplication (case-insensitive) | column?, keep? (first/last) |
| nlp.text_to_paragraphs | split text by blank lines into paragraphs (new rows) | column? |
| nlp.text_count_pattern | count occurrences of a pattern per row, optionally filter | column?, pattern, filter? (bool) |
| nlp.text_summary_report | comprehensive text dataset report: stats, quality, recommendations (output_type=query) | column? |
| nlp.text_stratified_sample | stratified random sample maintaining label distribution | column? (label col), n? (default 100) |

### Analysis (smart, high-level — use for complex questions, comparisons, insights)
| id | what it does | params |
|----|-------------|--------|
| analysis.compare_extremes | compare rows with highest vs lowest value — side-by-side table + chart | column? (numeric col to compare by) |
| analysis.deep_profile | deep statistical profile of a column: distribution, outliers, quartiles, chart | column? |
| analysis.group_insights | compare stats across groups of a categorical column + box plot | column? (group col), value_column? (numeric) |
| analysis.anomaly_detect | detect anomalies using IQR/Z-score — flag outlier rows + scatter chart | column?, method? (iqr/zscore) |
| analysis.data_quality | comprehensive data quality scores per column + overall score + chart | (none) |
| analysis.correlation_insights | find top correlations with explanation + scatter plots | n? (default 10) |
| analysis.compare_columns | side-by-side comparison of two columns: stats + overlapping histogram | columns (list of 2) |
| analysis.trend_detect | detect trends: direction, slope, moving average + chart | column?, window? (default 5) |
| analysis.segment_analysis | auto-segment data into quantile groups and describe each | column?, n? (segments, default 4) |
| analysis.auto_eda | automated EDA: key findings, quality issues, recommendations | (none) |
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
  - Binning/cutting with custom labels (pd.cut with formatting)
  - Custom calculations (percentages, ratios, derived metrics)
  - Pivot, melt, reshape, merge, join
  - Moving average, rolling window, cumulative operations
  - Complex multi-condition filtering
  - Any custom math/logic not covered by a handler

### output_type
  - "query" → stats, viz, questions, any read-only operation
  - "generate" → cleaning, transforms, data generation — anything that creates/modifies data

### Chart selection — THINK before picking a chart
When the user asks for a visualization, REASON about:
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
11. If user speaks Thai, understand the intent and plan normally — do NOT translate.
12. If the message is NOT about the dataset at all → use direct_answer (see format below).

## OUTPUT FORMAT — valid JSON, no markdown fences, no explanation

For direct_answer (NOT about the dataset):
{{
  "understanding": "one sentence",
  "output_type": "text",
  "direct_answer": true,
  "steps": []
}}

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

User: "plot how many of each bedroom" (count intent → bar chart)
{{"understanding":"Count of each bedroom value","output_type":"query","steps":[{{"step_num":1,"description":"Bar chart count of bedrooms","handler":{{"id":"viz.bar_chart","params":{{"column":"bedrooms"}}}}}}]}}

User: "show percentage of bedrooms" (percent intent → pie chart)
{{"understanding":"Percentage breakdown of bedrooms","output_type":"query","steps":[{{"step_num":1,"description":"Pie chart of bedroom percentage","handler":{{"id":"viz.pie_chart","params":{{"column":"bedrooms"}}}}}}]}}

User: "show distribution of price" (distribution intent → histogram)
{{"understanding":"Price distribution","output_type":"query","steps":[{{"step_num":1,"description":"Distribution of price","handler":{{"id":"viz.distribution","params":{{"column":"price"}}}}}}]}}

User: "plot price vs area" (relationship intent → scatter)
{{"understanding":"Scatter plot of price vs area","output_type":"query","steps":[{{"step_num":1,"description":"Scatter price vs area","handler":{{"id":"viz.scatter","params":{{"columns":["area","price"]}}}}}}]}}

User: "ไก่ ไข่ ไก่ หมา คำไหนมีมากที่สุด" (general question, NOT about dataset)
{{"understanding":"General question about word frequency — not related to the dataset","output_type":"text","direct_answer":true,"steps":[]}}

User: "what is machine learning?" (general knowledge)
{{"understanding":"General knowledge question","output_type":"text","direct_answer":true,"steps":[]}}

User: "thank you" (casual chat)
{{"understanding":"Casual response","output_type":"text","direct_answer":true,"steps":[]}}

User: "1+1 เท่ากับเท่าไหร่" (simple math, not about data)
{{"understanding":"Simple math question","output_type":"text","direct_answer":true,"steps":[]}}

User: "how do I write a for loop?" (coding help)
{{"understanding":"Programming question","output_type":"text","direct_answer":true,"steps":[]}}

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

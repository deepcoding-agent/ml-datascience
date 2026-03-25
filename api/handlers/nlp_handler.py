"""NLP / Text preprocessing handler — 45 handlers for text cleaning,
tokenization, vectorization, and NLP feature engineering."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px

from api.handlers.base import BaseHandler, HandlerResult
from api.handlers.viz_handler import _style
from api.logger import get_logger

log = get_logger(__name__)

# ── Built-in English stopwords (no NLTK dependency) ─────────────────────────

ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn",
    "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
})

# ── Basic sentiment lexicon (no external dependency) ────────────────────────

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "awesome", "outstanding", "superb", "brilliant", "love", "happy",
    "best", "perfect", "beautiful", "nice", "positive", "enjoy",
    "pleased", "satisfied", "recommend", "impressive", "delightful",
    "exceptional", "magnificent", "terrific", "fabulous", "marvelous",
    "like", "thank", "thanks", "helpful", "useful", "easy", "fast",
    "reliable", "efficient", "comfortable", "friendly", "smooth",
    "worth", "glad", "fortunate", "success", "win", "top", "favorite",
})

_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "horrible", "awful", "poor", "worst", "hate",
    "ugly", "disappointing", "disappointed", "negative", "boring",
    "slow", "broken", "useless", "waste", "annoying", "frustrating",
    "angry", "sad", "problem", "issue", "error", "fail", "failure",
    "wrong", "difficult", "hard", "expensive", "cheap", "rude",
    "terrible", "lousy", "dreadful", "inferior", "mediocre", "weak",
    "damage", "risk", "danger", "complaint", "regret", "unfortunately",
    "never", "worse", "painful", "unhappy", "unfair", "unreliable",
})

# ── Regex patterns for common extractions ───────────────────────────────────

_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "url": r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    "hashtag": r"#\w+",
    "mention": r"@\w+",
    "phone": r"\+?\d[\d\-\s()]{7,}\d",
    "number": r"-?\d+\.?\d*",
}

# ── Unicode range → language mapping ────────────────────────────────────────

_LANG_RANGES: list[tuple[int, int, str]] = [
    (0x0E00, 0x0E7F, "thai"),
    (0x4E00, 0x9FFF, "chinese"),
    (0x3040, 0x30FF, "japanese"),
    (0xAC00, 0xD7AF, "korean"),
    (0x0400, 0x04FF, "russian"),
    (0x0600, 0x06FF, "arabic"),
    (0x0900, 0x097F, "hindi"),
    (0x0041, 0x007A, "latin"),
]

# ── Common English suffixes for basic stemming ──────────────────────────────

_SUFFIX_RULES: list[tuple[str, str]] = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
    ("anci", "ance"), ("izer", "ize"), ("alli", "al"),
    ("entli", "ent"), ("eli", "e"), ("ousli", "ous"),
    ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
    ("biliti", "ble"), ("ingness", "ing"), ("lessly", "less"),
    ("ingly", "ing"), ("ness", ""), ("ment", ""),
    ("ing", ""), ("tion", "t"), ("sion", "s"),
    ("able", ""), ("ible", ""), ("ful", ""),
    ("less", ""), ("ly", ""), ("ed", ""), ("er", ""),
    ("es", ""), ("s", ""),
]


def _basic_stem(word: str) -> str:
    """Very basic suffix-stripping stemmer (no NLTK needed)."""
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(replacement) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def _get_text_cols(df: pd.DataFrame, col: str | None) -> list[str]:
    """Resolve target text columns — single specified or all object cols."""
    if col and col in df.columns:
        return [col]
    return df.select_dtypes(include="object").columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════
#  NLP Handler — 15 handlers
# ═══════════════════════════════════════════════════════════════════════════


class NlpHandler(BaseHandler):

    # ── 1. Text clean ─────────────────────────────────────────────────────

    @staticmethod
    def handle_text_clean(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Clean text: lowercase, remove HTML/URLs/emails/punctuation/numbers,
        normalize whitespace. Use strategy param to select specific steps."""
        col = params.get("column")
        strategy = params.get("strategy", "all")  # all|lowercase|no_punct|no_numbers|no_html|no_urls|no_emails
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        steps_applied: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)

            if strategy in ("all", "no_html"):
                s = s.str.replace(r"<[^>]+>", " ", regex=True)
                if "html" not in steps_applied:
                    steps_applied.append("remove_html")

            if strategy in ("all", "no_urls"):
                s = s.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
                if "urls" not in steps_applied:
                    steps_applied.append("remove_urls")

            if strategy in ("all", "no_emails"):
                s = s.str.replace(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", " ", regex=True)
                if "emails" not in steps_applied:
                    steps_applied.append("remove_emails")

            if strategy in ("all", "no_numbers"):
                s = s.str.replace(r"\d+", " ", regex=True)
                if "numbers" not in steps_applied:
                    steps_applied.append("remove_numbers")

            if strategy in ("all", "no_punct"):
                s = s.str.replace(r"[^\w\s]", " ", regex=True)
                if "punct" not in steps_applied:
                    steps_applied.append("remove_punctuation")

            if strategy in ("all", "lowercase"):
                s = s.str.lower()
                if "lowercase" not in steps_applied:
                    steps_applied.append("lowercase")

            # Always normalize whitespace
            s = s.str.replace(r"\s+", " ", regex=True).str.strip()
            result[c] = s

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Cleaned {len(text_cols)} text column(s): {', '.join(steps_applied)}",
        )

    # ── 2. Remove stopwords ───────────────────────────────────────────────

    @staticmethod
    def handle_remove_stopwords(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove English stopwords from text columns."""
        col = params.get("column")
        extra_stops = set(params.get("extra_words", []))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        stops = ENGLISH_STOPWORDS | extra_stops

        for c in text_cols:
            s = result[c].fillna("").astype(str)
            result[c] = s.apply(
                lambda t: " ".join(w for w in t.split() if w.lower() not in stops)
            )

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Removed stopwords from {len(text_cols)} column(s) ({len(stops)} stopwords)",
        )

    # ── 3. Tokenize ──────────────────────────────────────────────────────

    @staticmethod
    def handle_tokenize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Tokenize text into words using regex word boundaries.
        Creates {col}_tokens (list) and {col}_token_count columns."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            tokens = s.str.findall(r"\b\w+\b")
            result[f"{c}_tokens"] = tokens.apply(lambda x: ",".join(x) if x else "")
            result[f"{c}_token_count"] = tokens.str.len().fillna(0).astype(int)
            created.extend([f"{c}_tokens", f"{c}_token_count"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Tokenized {len(text_cols)} column(s) → {len(created)} new columns",
        )

    # ── 4. TF-IDF vectorization ──────────────────────────────────────────

    @staticmethod
    def handle_tfidf(df: pd.DataFrame, params: dict) -> HandlerResult:
        """TF-IDF vectorization — creates top-N feature columns from text."""
        col = params.get("column")
        max_features = int(params.get("n", 50))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            total_features = 0
            for c in text_cols:
                corpus = result[c].fillna("").astype(str)
                vec = TfidfVectorizer(max_features=max_features, stop_words="english")
                matrix = vec.fit_transform(corpus)
                feature_names = [f"{c}_tfidf_{w}" for w in vec.get_feature_names_out()]
                tfidf_df = pd.DataFrame(
                    matrix.toarray(), columns=feature_names, index=result.index,
                )
                result = pd.concat([result, tfidf_df], axis=1)
                total_features += len(feature_names)

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"TF-IDF: created {total_features} features from {len(text_cols)} column(s)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"TF-IDF error: {e}")

    # ── 5. Bag of Words (CountVectorizer) ────────────────────────────────

    @staticmethod
    def handle_bow(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Bag of Words — count-based vectorization of text columns."""
        col = params.get("column")
        max_features = int(params.get("n", 50))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import CountVectorizer

            total_features = 0
            for c in text_cols:
                corpus = result[c].fillna("").astype(str)
                vec = CountVectorizer(max_features=max_features, stop_words="english")
                matrix = vec.fit_transform(corpus)
                feature_names = [f"{c}_bow_{w}" for w in vec.get_feature_names_out()]
                bow_df = pd.DataFrame(
                    matrix.toarray(), columns=feature_names, index=result.index,
                )
                result = pd.concat([result, bow_df], axis=1)
                total_features += len(feature_names)

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Bag of Words: created {total_features} features from {len(text_cols)} column(s)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Bag of Words error: {e}")

    # ── 6. N-gram features ───────────────────────────────────────────────

    @staticmethod
    def handle_ngrams(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract word n-gram features using TF-IDF with n-gram range."""
        col = params.get("column")
        n = int(params.get("n", 2))  # bigrams by default
        max_features = int(params.get("max_features", 30))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            total_features = 0
            for c in text_cols:
                corpus = result[c].fillna("").astype(str)
                vec = TfidfVectorizer(
                    ngram_range=(n, n), max_features=max_features, stop_words="english",
                )
                matrix = vec.fit_transform(corpus)
                names = [f"{c}_{n}gram_{w.replace(' ', '_')}" for w in vec.get_feature_names_out()]
                ngram_df = pd.DataFrame(
                    matrix.toarray(), columns=names, index=result.index,
                )
                result = pd.concat([result, ngram_df], axis=1)
                total_features += len(names)

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"{n}-gram features: created {total_features} from {len(text_cols)} column(s)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"N-gram error: {e}")

    # ── 7. Regex extract ─────────────────────────────────────────────────

    @staticmethod
    def handle_regex_extract(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract patterns from text: email, url, hashtag, mention, phone, number, or custom regex."""
        col = params.get("column")
        pattern_name = params.get("pattern", "all")  # email|url|hashtag|mention|phone|number|all|custom
        custom_regex = params.get("regex")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        patterns: dict[str, str] = {}
        if custom_regex:
            patterns["custom"] = custom_regex
        elif pattern_name == "all":
            patterns = dict(_PATTERNS)
        elif pattern_name in _PATTERNS:
            patterns[pattern_name] = _PATTERNS[pattern_name]
        else:
            return HandlerResult(
                success=False,
                error=f"Unknown pattern '{pattern_name}'. Available: {list(_PATTERNS.keys())} or 'all'",
            )

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            for pname, regex in patterns.items():
                count_col = f"{c}_{pname}_count"
                result[count_col] = s.str.count(regex)
                match_col = f"{c}_{pname}_found"
                result[match_col] = s.str.findall(regex).apply(
                    lambda x: ",".join(x[:5]) if x else ""
                )
                created.extend([count_col, match_col])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Regex extract: {len(created)} columns from {len(text_cols)} text column(s)",
        )

    # ── 8. Sentiment score ───────────────────────────────────────────────

    @staticmethod
    def handle_sentiment_score(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Basic lexicon-based sentiment scoring (positive/negative/compound).
        Uses built-in word lists — no external NLP library needed."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str).str.lower()
            words_series = s.str.findall(r"\b\w+\b")

            pos_counts = words_series.apply(lambda ws: sum(1 for w in ws if w in _POSITIVE_WORDS))
            neg_counts = words_series.apply(lambda ws: sum(1 for w in ws if w in _NEGATIVE_WORDS))
            total = words_series.str.len().replace(0, 1)

            result[f"{c}_sentiment_pos"] = (pos_counts / total).round(4)
            result[f"{c}_sentiment_neg"] = (neg_counts / total).round(4)
            result[f"{c}_sentiment_compound"] = ((pos_counts - neg_counts) / total).round(4)
            result[f"{c}_sentiment_label"] = np.where(
                result[f"{c}_sentiment_compound"] > 0.05, "positive",
                np.where(result[f"{c}_sentiment_compound"] < -0.05, "negative", "neutral"),
            )
            created.extend([
                f"{c}_sentiment_pos", f"{c}_sentiment_neg",
                f"{c}_sentiment_compound", f"{c}_sentiment_label",
            ])

        # Summary chart for first column
        charts: list[str] = []
        if text_cols:
            label_col = f"{text_cols[0]}_sentiment_label"
            if label_col in result.columns:
                counts = result[label_col].value_counts()
                fig = px.bar(
                    x=counts.index, y=counts.values,
                    color=counts.index,
                    color_discrete_map={"positive": "#2EC4B6", "neutral": "#86868B", "negative": "#E71D36"},
                )
                _style(fig, title=f"Sentiment Distribution — {text_cols[0]} (n={len(result)})")
                fig.update_layout(xaxis_title="Sentiment", yaxis_title="Count", showlegend=False)
                charts.append(fig.to_json())

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            charts_plotly=charts,
            summary=f"Sentiment scored {len(text_cols)} column(s): {len(created)} features created",
        )

    # ── 9. Word frequency ────────────────────────────────────────────────

    @staticmethod
    def handle_word_frequency(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze word frequency — returns top-N words + Plotly bar chart."""
        col = params.get("column")
        n = int(params.get("n", 20))
        remove_stops = params.get("remove_stopwords", True)
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        corpus = df[target].fillna("").astype(str).str.lower()
        all_words: list[str] = []
        for text in corpus:
            words = re.findall(r"\b\w+\b", text)
            if remove_stops:
                words = [w for w in words if w not in ENGLISH_STOPWORDS]
            all_words.extend(words)

        freq = Counter(all_words).most_common(n)
        freq_df = pd.DataFrame(freq, columns=["word", "count"])
        freq_df["percentage"] = (freq_df["count"] / max(len(all_words), 1) * 100).round(2)

        fig = px.bar(
            freq_df, x="count", y="word", orientation="h",
            text="count",
        )
        fig.update_traces(marker_color="#FB8C3C", textposition="outside")
        _style(fig, title=f"Top {n} Words — {target} ({len(all_words):,} total words)")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Frequency", yaxis_title="Word")

        return HandlerResult(
            success=True, result_df=freq_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Top {n} words from '{target}' ({len(all_words):,} total, {len(set(all_words)):,} unique)",
        )

    # ── 10. Text similarity ──────────────────────────────────────────────

    @staticmethod
    def handle_text_similarity(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute pairwise text similarity using TF-IDF + cosine similarity.
        Returns similarity matrix (first 50 rows for performance)."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            target = text_cols[0]
            sample = df.head(50)
            corpus = sample[target].fillna("").astype(str)
            vec = TfidfVectorizer(max_features=100, stop_words="english")
            matrix = vec.fit_transform(corpus)
            sim = cosine_similarity(matrix)
            sim_df = pd.DataFrame(
                sim.round(3),
                index=[f"doc_{i}" for i in range(len(sim))],
                columns=[f"doc_{i}" for i in range(len(sim))],
            )

            import plotly.graph_objects as go
            fig = go.Figure(data=go.Heatmap(
                z=sim, colorscale="YlOrRd",
                x=sim_df.columns.tolist(), y=sim_df.index.tolist(),
            ))
            _style(fig, title=f"Text Similarity (cosine) — {target} (n={len(sample)})")

            return HandlerResult(
                success=True, result_df=sim_df, output_type="query",
                charts_plotly=[fig.to_json()],
                summary=f"Cosine similarity matrix for '{target}' ({len(sample)} documents)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Text similarity error: {e}")

    # ── 11. Vocabulary stats ─────────────────────────────────────────────

    @staticmethod
    def handle_vocab_stats(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Vocabulary statistics: unique tokens, type-token ratio, avg word length,
        hapax legomena (words appearing only once), top words."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        rows: list[dict] = []
        for c in text_cols:
            corpus = df[c].fillna("").astype(str).str.lower()
            all_words: list[str] = []
            for text in corpus:
                all_words.extend(re.findall(r"\b\w+\b", text))

            total = len(all_words)
            unique = len(set(all_words))
            freq = Counter(all_words)
            hapax = sum(1 for w, cnt in freq.items() if cnt == 1)
            avg_len = np.mean([len(w) for w in all_words]) if all_words else 0

            rows.append({
                "column": c,
                "total_tokens": total,
                "unique_tokens": unique,
                "type_token_ratio": round(unique / max(total, 1), 4),
                "avg_word_length": round(avg_len, 2),
                "hapax_legomena": hapax,
                "hapax_ratio": round(hapax / max(unique, 1), 4),
                "top_5_words": ", ".join(w for w, _ in freq.most_common(5)),
                "docs_with_text": int((corpus.str.len() > 0).sum()),
                "empty_docs": int((corpus.str.len() == 0).sum()),
            })

        result_df = pd.DataFrame(rows)
        lines = []
        for r in rows:
            lines.append(
                f"**{r['column']}**: {r['total_tokens']:,} tokens, "
                f"{r['unique_tokens']:,} unique (TTR={r['type_token_ratio']}), "
                f"avg length={r['avg_word_length']}"
            )

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            summary="\n".join(lines),
        )

    # ── 12. Text normalize (stem + accent strip) ─────────────────────────

    @staticmethod
    def handle_text_normalize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Normalize text: strip accents, basic stemming (suffix removal),
        lowercase. Useful for reducing vocabulary before vectorization."""
        col = params.get("column")
        stem = params.get("stem", True)
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        def normalize_text(text: str) -> str:
            # Strip accents
            nfkd = unicodedata.normalize("NFKD", text)
            stripped = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
            # Lowercase
            stripped = stripped.lower()
            # Basic stemming
            if stem:
                words = stripped.split()
                words = [_basic_stem(w) for w in words]
                return " ".join(words)
            return stripped

        for c in text_cols:
            result[c] = result[c].fillna("").astype(str).apply(normalize_text)

        steps = ["accent_strip", "lowercase"]
        if stem:
            steps.append("basic_stemming")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Normalized {len(text_cols)} column(s): {', '.join(steps)}",
        )

    # ── 13. Language detect ──────────────────────────────────────────────

    @staticmethod
    def handle_language_detect(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Detect language per row based on Unicode character ranges.
        Creates {col}_language column."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        def detect_lang(text: str) -> str:
            if not text.strip():
                return "empty"
            char_counts: dict[str, int] = {}
            for ch in text:
                cp = ord(ch)
                for lo, hi, lang in _LANG_RANGES:
                    if lo <= cp <= hi:
                        char_counts[lang] = char_counts.get(lang, 0) + 1
                        break
            if not char_counts:
                return "unknown"
            return max(char_counts, key=char_counts.get)  # type: ignore[arg-type]

        created: list[str] = []
        for c in text_cols:
            lang_col = f"{c}_language"
            result[lang_col] = result[c].fillna("").astype(str).apply(detect_lang)
            created.append(lang_col)

        # Chart
        charts: list[str] = []
        if created:
            lang_counts = result[created[0]].value_counts()
            fig = px.pie(
                values=lang_counts.values, names=lang_counts.index,
            )
            _style(fig, title=f"Language Distribution — {text_cols[0]} (n={len(result)})")
            charts.append(fig.to_json())

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            charts_plotly=charts,
            summary=f"Detected language in {len(text_cols)} column(s): {', '.join(created)}",
        )

    # ── 14. Hash vectorize ───────────────────────────────────────────────

    @staticmethod
    def handle_hash_vectorize(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Feature hashing — fast, memory-efficient text vectorization.
        Creates N hashed feature columns (no vocabulary needed)."""
        col = params.get("column")
        n_features = int(params.get("n", 32))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import HashingVectorizer

            total = 0
            for c in text_cols:
                corpus = result[c].fillna("").astype(str)
                vec = HashingVectorizer(n_features=n_features, alternate_sign=False)
                matrix = vec.fit_transform(corpus)
                names = [f"{c}_hash_{i}" for i in range(n_features)]
                hash_df = pd.DataFrame(
                    matrix.toarray(), columns=names, index=result.index,
                )
                result = pd.concat([result, hash_df], axis=1)
                total += n_features

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Hash vectorization: {total} features from {len(text_cols)} column(s)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Hash vectorize error: {e}")

    # ── 15. Text to integer sequences ────────────────────────────────────

    @staticmethod
    def handle_text_encode(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Encode text as integer sequences (word → ID mapping).
        Creates {col}_encoded column with comma-separated word IDs
        and {col}_vocab_size with vocabulary size. Useful for deep learning input."""
        col = params.get("column")
        max_vocab = int(params.get("max_vocab", 5000))
        max_len = int(params.get("max_len", 100))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            corpus = result[c].fillna("").astype(str).str.lower()
            # Build vocabulary from all text
            all_words: list[str] = []
            for text in corpus:
                all_words.extend(re.findall(r"\b\w+\b", text))

            freq = Counter(all_words)
            # Reserve 0=pad, 1=unknown
            vocab = {word: idx + 2 for idx, (word, _) in enumerate(freq.most_common(max_vocab))}

            def encode_text(text: str) -> str:
                words = re.findall(r"\b\w+\b", text.lower())[:max_len]
                ids = [str(vocab.get(w, 1)) for w in words]
                # Pad to max_len
                ids.extend(["0"] * (max_len - len(ids)))
                return ",".join(ids)

            result[f"{c}_encoded"] = corpus.apply(encode_text)
            result[f"{c}_vocab_size"] = len(vocab) + 2  # +2 for pad and unknown
            created.extend([f"{c}_encoded", f"{c}_vocab_size"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=(
                f"Encoded {len(text_cols)} column(s) as integer sequences "
                f"(vocab={min(len(freq), max_vocab)+2}, max_len={max_len})"
            ),
            metadata={"vocab_size": min(len(freq), max_vocab) + 2, "max_len": max_len},
        )

    # ── 16. Keyword extraction (TF-IDF per document) ─────────────────────

    @staticmethod
    def handle_keyword_extract(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract top keywords per document using TF-IDF scores.
        Creates {col}_keywords column with comma-separated top words."""
        col = params.get("column")
        n_keywords = int(params.get("n", 5))
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            created: list[str] = []
            for c in text_cols:
                corpus = result[c].fillna("").astype(str)
                vec = TfidfVectorizer(max_features=500, stop_words="english")
                matrix = vec.fit_transform(corpus)
                feature_names = vec.get_feature_names_out()

                keywords: list[str] = []
                for i in range(matrix.shape[0]):
                    row = matrix[i].toarray().flatten()
                    top_idx = row.argsort()[-n_keywords:][::-1]
                    top_words = [feature_names[j] for j in top_idx if row[j] > 0]
                    keywords.append(", ".join(top_words))

                result[f"{c}_keywords"] = keywords
                created.append(f"{c}_keywords")

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Extracted top-{n_keywords} keywords from {len(text_cols)} column(s)",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Keyword extraction error: {e}")

    # ── 17. Character-level features ─────────────────────────────────────

    @staticmethod
    def handle_char_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract character-level features: punctuation count/ratio, digit ratio,
        uppercase ratio, special char ratio, whitespace ratio, avg word length."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            length = s.str.len().replace(0, 1)

            result[f"{c}_char_count"] = s.str.len()
            result[f"{c}_punct_count"] = s.str.count(r"[^\w\s]")
            result[f"{c}_punct_ratio"] = (result[f"{c}_punct_count"] / length).round(4)
            result[f"{c}_digit_ratio"] = (s.str.count(r"\d") / length).round(4)
            result[f"{c}_upper_ratio"] = (s.str.count(r"[A-Z]") / length).round(4)
            result[f"{c}_lower_ratio"] = (s.str.count(r"[a-z]") / length).round(4)
            result[f"{c}_space_ratio"] = (s.str.count(r"\s") / length).round(4)
            result[f"{c}_special_ratio"] = (s.str.count(r"[^a-zA-Z0-9\s]") / length).round(4)
            # Average word length
            words = s.str.findall(r"\b\w+\b")
            result[f"{c}_avg_word_len"] = words.apply(
                lambda ws: round(np.mean([len(w) for w in ws]), 2) if ws else 0.0
            )
            created.extend([
                f"{c}_char_count", f"{c}_punct_count", f"{c}_punct_ratio",
                f"{c}_digit_ratio", f"{c}_upper_ratio", f"{c}_lower_ratio",
                f"{c}_space_ratio", f"{c}_special_ratio", f"{c}_avg_word_len",
            ])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted {len(created)} character-level features from {len(text_cols)} column(s)",
        )

    # ── 18. Sentence-level features ──────────────────────────────────────

    @staticmethod
    def handle_sentence_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract sentence-level features: sentence count, avg/min/max sentence length,
        question/exclamation counts."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)

            def sent_stats(text: str) -> dict:
                sents = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
                n = len(sents)
                lens = [len(x.split()) for x in sents] if sents else [0]
                return {
                    "count": n,
                    "avg_len": round(np.mean(lens), 2),
                    "max_len": max(lens),
                    "min_len": min(lens),
                }

            stats = s.apply(sent_stats).apply(pd.Series)
            result[f"{c}_sent_count"] = stats["count"].astype(int)
            result[f"{c}_sent_avg_len"] = stats["avg_len"]
            result[f"{c}_sent_max_len"] = stats["max_len"].astype(int)
            result[f"{c}_sent_min_len"] = stats["min_len"].astype(int)
            result[f"{c}_question_count"] = s.str.count(r"\?")
            result[f"{c}_exclamation_count"] = s.str.count(r"!")

            created.extend([
                f"{c}_sent_count", f"{c}_sent_avg_len", f"{c}_sent_max_len",
                f"{c}_sent_min_len", f"{c}_question_count", f"{c}_exclamation_count",
            ])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted {len(created)} sentence features from {len(text_cols)} column(s)",
        )

    # ── 19. Readability score ────────────────────────────────────────────

    @staticmethod
    def handle_readability_score(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute readability metrics: Flesch Reading Ease (approx),
        Coleman-Liau Index, Automated Readability Index (ARI)."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        def _count_syllables(word: str) -> int:
            """Approximate syllable count using vowel groups."""
            word = word.lower().rstrip("e")
            vowels = re.findall(r"[aeiouy]+", word)
            return max(1, len(vowels))

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)

            def readability(text: str) -> dict:
                words_list = re.findall(r"\b\w+\b", text)
                sents = [x for x in re.split(r"[.!?]+", text) if x.strip()]
                n_words = max(len(words_list), 1)
                n_sents = max(len(sents), 1)
                n_chars = sum(len(w) for w in words_list)
                n_syllables = sum(_count_syllables(w) for w in words_list)
                # Flesch Reading Ease
                fre = 206.835 - 1.015 * (n_words / n_sents) - 84.6 * (n_syllables / n_words)
                # Coleman-Liau Index
                L = (n_chars / n_words) * 100
                S = (n_sents / n_words) * 100
                cli = 0.0588 * L - 0.296 * S - 15.8
                # Automated Readability Index
                ari = 4.71 * (n_chars / n_words) + 0.5 * (n_words / n_sents) - 21.43
                return {
                    "flesch": round(max(0, min(fre, 120)), 1),
                    "coleman_liau": round(cli, 1),
                    "ari": round(ari, 1),
                }

            scores = s.apply(readability).apply(pd.Series)
            result[f"{c}_flesch_score"] = scores["flesch"]
            result[f"{c}_coleman_liau"] = scores["coleman_liau"]
            result[f"{c}_ari_score"] = scores["ari"]
            created.extend([f"{c}_flesch_score", f"{c}_coleman_liau", f"{c}_ari_score"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Computed readability scores for {len(text_cols)} column(s): Flesch, Coleman-Liau, ARI",
        )

    # ── 20. Text deduplication ───────────────────────────────────────────

    @staticmethod
    def handle_text_dedup(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Find and remove near-duplicate texts using TF-IDF cosine similarity.
        Rows with similarity > threshold are flagged or removed."""
        col = params.get("column")
        threshold = float(params.get("threshold", 0.9))
        action = params.get("action", "flag")  # flag | remove
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            target = text_cols[0]
            result = df.copy()
            corpus = result[target].fillna("").astype(str)
            vec = TfidfVectorizer(max_features=200, stop_words="english")
            matrix = vec.fit_transform(corpus)
            sim = cosine_similarity(matrix)

            # Find duplicates (keep first occurrence)
            is_dup = np.zeros(len(result), dtype=bool)
            dup_of: list[int] = [-1] * len(result)
            for i in range(len(sim)):
                if is_dup[i]:
                    continue
                for j in range(i + 1, len(sim)):
                    if not is_dup[j] and sim[i][j] >= threshold:
                        is_dup[j] = True
                        dup_of[j] = i

            n_dups = int(is_dup.sum())
            result["_text_is_duplicate"] = is_dup
            result["_text_duplicate_of"] = dup_of

            if action == "remove" and n_dups > 0:
                result = result[~result["_text_is_duplicate"]].drop(
                    columns=["_text_is_duplicate", "_text_duplicate_of"],
                ).reset_index(drop=True)
                return HandlerResult(
                    success=True, result_df=result, output_type="generate",
                    summary=f"Removed {n_dups} near-duplicate texts (threshold={threshold}), {len(result)} rows remain",
                )

            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Flagged {n_dups} near-duplicate texts out of {len(result)} (threshold={threshold})",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Text dedup error: {e}")

    # ── 21. Emoji / emoticon features ────────────────────────────────────

    @staticmethod
    def handle_emoji_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract emoji and emoticon features: count, ratio, list of emojis found.
        Detects Unicode emojis and common text emoticons."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        # Unicode emoji pattern (covers most common emoji ranges)
        emoji_pattern = re.compile(
            "[\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"   # symbols & pictographs
            "\U0001F680-\U0001F6FF"   # transport & map
            "\U0001F1E0-\U0001F1FF"   # flags
            "\U00002702-\U000027B0"   # dingbats
            "\U000024C2-\U0001F251"   # misc
            "\U0001F900-\U0001F9FF"   # supplemental
            "\U0001FA00-\U0001FA6F"   # chess symbols
            "\U0001FA70-\U0001FAFF"   # symbols extended
            "]+", flags=re.UNICODE,
        )
        # Common text emoticons
        emoticon_pattern = re.compile(
            r"(?:[:;=][-']?[)(DPp/\\|Oo3><])|(?:[)(DPp><][-']?[:;=])|"
            r"<3|</3|:'\(|:\)|;\)|:D|:P|:O|xD|XD|:-\)|:-\(|:\(|T_T|>_<|\^_\^"
        )

        created: list[str] = []
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            # Emoji counts
            result[f"{c}_emoji_count"] = s.apply(lambda t: len(emoji_pattern.findall(t)))
            result[f"{c}_emoticon_count"] = s.apply(lambda t: len(emoticon_pattern.findall(t)))
            result[f"{c}_emoji_total"] = result[f"{c}_emoji_count"] + result[f"{c}_emoticon_count"]
            char_len = s.str.len().replace(0, 1)
            result[f"{c}_emoji_ratio"] = (result[f"{c}_emoji_total"] / char_len).round(4)
            # List emojis found
            result[f"{c}_emojis_found"] = s.apply(
                lambda t: ",".join(emoji_pattern.findall(t)[:10])
            )
            created.extend([
                f"{c}_emoji_count", f"{c}_emoticon_count", f"{c}_emoji_total",
                f"{c}_emoji_ratio", f"{c}_emojis_found",
            ])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Extracted {len(created)} emoji features from {len(text_cols)} column(s)",
        )

    # ── 22. PII masking / anonymization ──────────────────────────────────

    @staticmethod
    def handle_text_mask_pii(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Mask personally identifiable information: emails, phone numbers,
        credit card numbers, SSN-like patterns, IP addresses."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        pii_patterns: list[tuple[str, str, str]] = [
            ("email", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
            ("phone", r"\+?\d[\d\-\s()]{7,}\d", "[PHONE]"),
            ("credit_card", r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[CREDIT_CARD]"),
            ("ssn", r"\b\d{3}[\s\-]?\d{2}[\s\-]?\d{4}\b", "[SSN]"),
            ("ip_address", r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP]"),
            ("url", r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", "[URL]"),
        ]

        total_masked = 0
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            for pii_name, pattern, replacement in pii_patterns:
                count_before = s.str.count(pattern).sum()
                s = s.str.replace(pattern, replacement, regex=True)
                total_masked += int(count_before)
            result[c] = s

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Masked {total_masked} PII instances across {len(text_cols)} column(s): email, phone, credit card, SSN, IP, URL",
        )

    # ── 23. Text augmentation ────────────────────────────────────────────

    @staticmethod
    def handle_text_augment(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Simple text augmentation: random word deletion, swap, and synonym-free duplication.
        Creates augmented copies appended to the dataset. Useful for balancing NLP training sets."""
        col = params.get("column")
        n_aug = int(params.get("n", 1))  # number of augmented copies per row
        strategy = params.get("strategy", "mixed")  # delete | swap | mixed
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        rng = np.random.RandomState(42)
        target = text_cols[0]

        def augment_one(text: str, method: str) -> str:
            words = text.split()
            if len(words) < 3:
                return text
            if method == "delete":
                idx = rng.randint(0, len(words))
                return " ".join(w for i, w in enumerate(words) if i != idx)
            elif method == "swap":
                i = rng.randint(0, max(len(words) - 1, 1))
                j = min(i + 1, len(words) - 1)
                words[i], words[j] = words[j], words[i]
                return " ".join(words)
            else:  # mixed
                method = rng.choice(["delete", "swap"])
                return augment_one(text, method)

        result = df.copy()
        aug_rows: list[pd.DataFrame] = []
        for _ in range(n_aug):
            aug = result.copy()
            aug[target] = aug[target].fillna("").astype(str).apply(
                lambda t: augment_one(t, strategy)
            )
            aug_rows.append(aug)

        result = pd.concat([result] + aug_rows, ignore_index=True)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Augmented '{target}': {len(df)} → {len(result)} rows ({n_aug}x augmentation, strategy={strategy})",
        )

    # ── 24. Collocations (significant word pairs) ────────────────────────

    @staticmethod
    def handle_collocations(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Find significant word collocations (bigrams) using frequency and PMI.
        Returns top-N collocations with scores + bar chart."""
        col = params.get("column")
        n = int(params.get("n", 20))
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        corpus = df[target].fillna("").astype(str).str.lower()

        # Collect bigrams and unigram counts
        bigram_counts: Counter = Counter()
        unigram_counts: Counter = Counter()
        total_bigrams = 0

        for text in corpus:
            words = [w for w in re.findall(r"\b\w+\b", text) if w not in ENGLISH_STOPWORDS and len(w) > 1]
            unigram_counts.update(words)
            for i in range(len(words) - 1):
                bigram_counts[(words[i], words[i + 1])] += 1
                total_bigrams += 1

        if total_bigrams == 0:
            return HandlerResult(success=False, error="No collocations found — text may be too short")

        total_unigrams = sum(unigram_counts.values())

        # Compute PMI for top bigrams
        rows: list[dict] = []
        for (w1, w2), count in bigram_counts.most_common(n * 3):
            if count < 2:
                continue
            p_bi = count / total_bigrams
            p_w1 = unigram_counts[w1] / total_unigrams
            p_w2 = unigram_counts[w2] / total_unigrams
            pmi = np.log2(p_bi / max(p_w1 * p_w2, 1e-10))
            rows.append({
                "collocation": f"{w1} {w2}",
                "frequency": count,
                "pmi": round(pmi, 3),
            })

        rows.sort(key=lambda r: r["pmi"], reverse=True)
        result_df = pd.DataFrame(rows[:n])

        if not result_df.empty:
            fig = px.bar(
                result_df, x="pmi", y="collocation", orientation="h",
                text="frequency",
            )
            fig.update_traces(marker_color="#2EC4B6", textposition="outside")
            _style(fig, title=f"Top {n} Collocations — {target} (PMI-ranked)")
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="PMI Score", yaxis_title="Word Pair",
            )
            charts = [fig.to_json()]
        else:
            charts = []

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=charts,
            summary=f"Found {len(result_df)} significant collocations in '{target}'",
        )

    # ── 25. Word cloud data + treemap visualization ──────────────────────

    @staticmethod
    def handle_word_cloud(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Generate word cloud data (word + weight) with Plotly treemap visualization."""
        col = params.get("column")
        n = int(params.get("n", 40))
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        corpus = df[target].fillna("").astype(str).str.lower()
        all_words: list[str] = []
        for text in corpus:
            words = [w for w in re.findall(r"\b\w+\b", text) if w not in ENGLISH_STOPWORDS and len(w) > 2]
            all_words.extend(words)

        freq = Counter(all_words).most_common(n)
        if not freq:
            return HandlerResult(success=False, error="No words found after filtering")

        cloud_df = pd.DataFrame(freq, columns=["word", "weight"])
        cloud_df["percentage"] = (cloud_df["weight"] / max(sum(c for _, c in freq), 1) * 100).round(2)

        fig = px.treemap(
            cloud_df, path=["word"], values="weight",
            color="weight", color_continuous_scale="YlOrRd",
        )
        _style(fig, title=f"Word Cloud — {target} (top {n} words)")
        fig.update_traces(textinfo="label+value")

        return HandlerResult(
            success=True, result_df=cloud_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Word cloud for '{target}': top {n} words from {len(all_words):,} total",
        )

    # ── 26. Text filter (by length, keyword, language) ───────────────────

    @staticmethod
    def handle_text_filter(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Filter rows by text criteria: min/max length, contains keyword,
        min word count. Useful for cleaning out empty or too-short texts."""
        col = params.get("column")
        min_len = params.get("min_len")          # minimum character length
        max_len = params.get("max_len")          # maximum character length
        min_words = params.get("min_words")      # minimum word count
        contains = params.get("contains")        # must contain keyword
        not_contains = params.get("not_contains") # must not contain keyword
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        result = df.copy()
        s = result[target].fillna("").astype(str)
        original = len(result)
        mask = pd.Series(True, index=result.index)

        if min_len is not None:
            mask &= s.str.len() >= int(min_len)
        if max_len is not None:
            mask &= s.str.len() <= int(max_len)
        if min_words is not None:
            mask &= s.str.split().str.len().fillna(0) >= int(min_words)
        if contains is not None:
            mask &= s.str.contains(str(contains), case=False, na=False)
        if not_contains is not None:
            mask &= ~s.str.contains(str(not_contains), case=False, na=False)

        result = result[mask].reset_index(drop=True)
        removed = original - len(result)

        filters = []
        if min_len is not None: filters.append(f"min_len={min_len}")
        if max_len is not None: filters.append(f"max_len={max_len}")
        if min_words is not None: filters.append(f"min_words={min_words}")
        if contains is not None: filters.append(f"contains='{contains}'")
        if not_contains is not None: filters.append(f"not_contains='{not_contains}'")

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Filtered '{target}': {original} → {len(result)} rows (removed {removed}), filters: {', '.join(filters)}",
        )

    # ── 27. Text class balance check ─────────────────────────────────────

    @staticmethod
    def handle_class_balance_text(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze text label/category class balance with visualization.
        Shows class distribution, imbalance ratio, and avg text length per class."""
        col = params.get("column")
        text_col = params.get("text_column")
        if not col or col not in df.columns:
            # Auto-detect: find low-cardinality string column
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            candidates = [c for c in cat_cols if df[c].nunique() <= 50]
            if not candidates:
                return HandlerResult(success=False, error="No categorical/label column found. Specify column= param.")
            col = candidates[0]

        counts = df[col].value_counts()
        imbalance_ratio = round(counts.max() / max(counts.min(), 1), 2)

        rows: list[dict] = []
        for label, count in counts.items():
            row = {"label": str(label), "count": count, "percentage": round(count / len(df) * 100, 2)}
            # If text column provided, compute avg length per class
            if text_col and text_col in df.columns:
                subset = df[df[col] == label][text_col].fillna("").astype(str)
                row["avg_text_length"] = round(subset.str.len().mean(), 1)
                row["avg_word_count"] = round(subset.str.split().str.len().mean(), 1)
            rows.append(row)

        result_df = pd.DataFrame(rows)

        fig = px.bar(
            result_df, x="label", y="count", color="label", text="count",
        )
        fig.update_traces(textposition="outside")
        _style(fig, title=f"Class Balance — {col} (imbalance ratio: {imbalance_ratio}x, n={len(df)})")
        fig.update_layout(xaxis_title="Class", yaxis_title="Count", showlegend=False)

        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"Class balance for '{col}': {len(counts)} classes, imbalance ratio={imbalance_ratio}x, majority='{counts.index[0]}' ({counts.iloc[0]})",
        )

    # ── 28. Text chunking ────────────────────────────────────────────────

    @staticmethod
    def handle_text_chunk(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split long texts into fixed-size chunks (by word count).
        Creates new rows for each chunk — useful for processing long documents."""
        col = params.get("column")
        chunk_size = int(params.get("chunk_size", 200))  # words per chunk
        overlap = int(params.get("overlap", 20))
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        chunks: list[dict] = []

        for idx, row in df.iterrows():
            text = str(row.get(target, ""))
            words = text.split()

            if len(words) <= chunk_size:
                chunk_row = row.to_dict()
                chunk_row["_chunk_id"] = 0
                chunk_row["_chunk_total"] = 1
                chunks.append(chunk_row)
            else:
                step = max(chunk_size - overlap, 1)
                total_chunks = max(1, (len(words) - overlap) // step + (1 if (len(words) - overlap) % step else 0))
                for i, start in enumerate(range(0, len(words), step)):
                    chunk_words = words[start : start + chunk_size]
                    if not chunk_words:
                        break
                    chunk_row = row.to_dict()
                    chunk_row[target] = " ".join(chunk_words)
                    chunk_row["_chunk_id"] = i
                    chunk_row["_chunk_total"] = total_chunks
                    chunks.append(chunk_row)

        result = pd.DataFrame(chunks).reset_index(drop=True)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Chunked '{target}': {len(df)} docs → {len(result)} chunks (size={chunk_size}, overlap={overlap})",
        )

    # ── 29. Spelling / OOV features ──────────────────────────────────────

    @staticmethod
    def handle_spelling_features(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Estimate out-of-vocabulary (OOV) word ratio as a proxy for spelling quality.
        Builds a vocabulary from the corpus; words appearing only once are likely misspelled.
        Creates oov_count and oov_ratio columns."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        created: list[str] = []
        for c in text_cols:
            corpus = result[c].fillna("").astype(str).str.lower()
            # Build corpus vocabulary
            all_words: list[str] = []
            for text in corpus:
                all_words.extend(re.findall(r"\b[a-zA-Z]+\b", text))

            freq = Counter(all_words)
            # Words appearing only once with length > 2 are likely misspelled
            rare = {w for w, cnt in freq.items() if cnt == 1 and len(w) > 2}

            def oov_stats(text: str) -> tuple[int, float]:
                words = [w for w in re.findall(r"\b[a-zA-Z]+\b", text.lower()) if len(w) > 2]
                if not words:
                    return 0, 0.0
                oov = sum(1 for w in words if w in rare)
                return oov, round(oov / len(words), 4)

            stats = corpus.apply(oov_stats)
            result[f"{c}_oov_count"] = stats.apply(lambda x: x[0])
            result[f"{c}_oov_ratio"] = stats.apply(lambda x: x[1])
            created.extend([f"{c}_oov_count", f"{c}_oov_ratio"])

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Computed OOV/spelling features for {len(text_cols)} column(s): {len(rare)} rare words detected",
        )

    # ── 30. Text column concatenation ────────────────────────────────────

    @staticmethod
    def handle_text_concat(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Combine multiple text columns into a single corpus column.
        Useful before vectorization when text is split across multiple fields."""
        columns = params.get("columns")
        separator = params.get("separator", " ")
        new_name = params.get("new_name", "text_combined")
        result = df.copy()

        if columns and isinstance(columns, list):
            text_cols = [c for c in columns if c in result.columns]
        else:
            text_cols = result.select_dtypes(include="object").columns.tolist()

        if len(text_cols) < 2:
            return HandlerResult(
                success=False,
                error=f"Need at least 2 text columns to concatenate. Found: {text_cols}",
            )

        result[new_name] = result[text_cols].fillna("").astype(str).agg(separator.join, axis=1)
        result[new_name] = result[new_name].str.strip()

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Concatenated {len(text_cols)} columns → '{new_name}': {', '.join(text_cols)}",
        )

    # ── 31. Text replace (find & replace with regex) ─────────────────────

    @staticmethod
    def handle_text_replace(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Find and replace text patterns (regex or literal) across text columns.
        Supports multiple replacements via a mapping dict."""
        col = params.get("column")
        pattern = params.get("pattern", "")
        replacement = params.get("replacement", "")
        mapping = params.get("mapping")  # dict of {find: replace, ...}
        use_regex = params.get("regex", True)
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        total_replacements = 0
        for c in text_cols:
            s = result[c].fillna("").astype(str)
            if mapping and isinstance(mapping, dict):
                for find, repl in mapping.items():
                    count = s.str.count(find).sum()
                    total_replacements += int(count)
                    s = s.str.replace(find, str(repl), regex=bool(use_regex))
            elif pattern:
                count = s.str.count(pattern).sum()
                total_replacements += int(count)
                s = s.str.replace(pattern, replacement, regex=bool(use_regex))
            result[c] = s

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Replaced {total_replacements} occurrences across {len(text_cols)} column(s)",
        )

    # ── 32. Split text into sentences (new rows) ────────────────────────

    @staticmethod
    def handle_text_split_sentences(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split text into individual sentences. Each sentence becomes a new row.
        Useful for sentence-level classification or analysis."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        rows: list[dict] = []

        for _, row in df.iterrows():
            text = str(row.get(target, ""))
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sents:
                sents = [text]
            for i, sent in enumerate(sents):
                new_row = row.to_dict()
                new_row[target] = sent
                new_row["_sentence_id"] = i
                new_row["_sentence_total"] = len(sents)
                rows.append(new_row)

        result = pd.DataFrame(rows).reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Split '{target}': {len(df)} docs → {len(result)} sentences",
        )

    # ── 33. Text oversample (balance minority classes) ───────────────────

    @staticmethod
    def handle_text_oversample(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Oversample minority text classes to balance the dataset.
        Duplicates rows from minority classes to match the majority class count."""
        label_col = params.get("column")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not label_col or label_col not in df.columns:
            candidates = [c for c in cat_cols if df[c].nunique() <= 20]
            label_col = candidates[0] if candidates else None
        if label_col is None:
            return HandlerResult(success=False, error="No label column found for oversampling")

        counts = df[label_col].value_counts()
        max_count = counts.max()
        parts: list[pd.DataFrame] = []
        for label, count in counts.items():
            subset = df[df[label_col] == label]
            if count < max_count:
                repeat = max_count // count
                remainder = max_count % count
                oversampled = pd.concat([subset] * repeat + [subset.head(remainder)], ignore_index=True)
                parts.append(oversampled)
            else:
                parts.append(subset)

        result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Oversampled '{label_col}': {len(df)} → {len(result)} rows (all classes now ≈{max_count})",
        )

    # ── 34. Document-term matrix ─────────────────────────────────────────

    @staticmethod
    def handle_doc_term_matrix(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Build a full document-term frequency matrix.
        Returns a sparse-style DataFrame with word counts per document."""
        col = params.get("column")
        max_features = int(params.get("n", 100))
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        try:
            from sklearn.feature_extraction.text import CountVectorizer
            target = text_cols[0]
            corpus = df[target].fillna("").astype(str)
            vec = CountVectorizer(max_features=max_features, stop_words="english")
            matrix = vec.fit_transform(corpus)
            dtm = pd.DataFrame(
                matrix.toarray(),
                columns=vec.get_feature_names_out(),
                index=df.index,
            )
            return HandlerResult(
                success=True, result_df=dtm, output_type="generate",
                summary=f"Document-term matrix: {dtm.shape[0]} docs × {dtm.shape[1]} terms",
            )
        except Exception as e:
            return HandlerResult(success=False, error=f"Doc-term matrix error: {e}")

    # ── 35. Sliding window context extraction ────────────────────────────

    @staticmethod
    def handle_text_window(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract sliding window contexts around a target keyword.
        Creates new rows with the keyword and its surrounding context."""
        col = params.get("column")
        keyword = params.get("keyword", "")
        window_size = int(params.get("window", 5))
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")
        if not keyword:
            return HandlerResult(success=False, error="Specify keyword= parameter")

        target = text_cols[0]
        contexts: list[dict] = []
        kw_lower = keyword.lower()

        for idx, row in df.iterrows():
            text = str(row.get(target, ""))
            words = text.split()
            for i, w in enumerate(words):
                if kw_lower in w.lower():
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    ctx = " ".join(words[start:end])
                    contexts.append({
                        "source_row": int(idx),  # type: ignore[arg-type]
                        "keyword": keyword,
                        "position": i,
                        "context": ctx,
                    })

        if not contexts:
            return HandlerResult(success=False, error=f"Keyword '{keyword}' not found in any text")

        result = pd.DataFrame(contexts)
        return HandlerResult(
            success=True, result_df=result, output_type="query",
            summary=f"Found {len(contexts)} occurrences of '{keyword}' with ±{window_size} word window",
        )

    # ── 36. Label from keyword rules ─────────────────────────────────────

    @staticmethod
    def handle_text_label_rules(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Create text labels based on keyword rules.
        mapping: dict of {label: [keyword1, keyword2, ...]}."""
        col = params.get("column")
        mapping = params.get("mapping")  # {"positive": ["good","great"], "negative": ["bad","poor"]}
        default_label = params.get("default", "other")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")
        if not mapping or not isinstance(mapping, dict):
            return HandlerResult(success=False, error="Provide mapping= dict: {label: [keywords]}")

        target = text_cols[0]
        result = df.copy()
        s = result[target].fillna("").astype(str).str.lower()

        def classify(text: str) -> str:
            for label, keywords in mapping.items():
                for kw in keywords:
                    if kw.lower() in text:
                        return str(label)
            return default_label

        result[f"{target}_label"] = s.apply(classify)
        dist = result[f"{target}_label"].value_counts().to_dict()

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Labeled {len(result)} rows using {len(mapping)} rules. Distribution: {dist}",
        )

    # ── 37. Word overlap / Jaccard similarity ────────────────────────────

    @staticmethod
    def handle_word_overlap(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Compute word overlap (Jaccard similarity) between two text columns
        or between consecutive rows of the same column."""
        columns = params.get("columns", [])
        col = params.get("column")
        result = df.copy()

        if columns and len(columns) >= 2 and all(c in df.columns for c in columns[:2]):
            c1, c2 = columns[0], columns[1]
            s1 = result[c1].fillna("").astype(str).str.lower()
            s2 = result[c2].fillna("").astype(str).str.lower()

            def jaccard(a: str, b: str) -> float:
                wa = set(re.findall(r"\b\w+\b", a))
                wb = set(re.findall(r"\b\w+\b", b))
                if not wa and not wb:
                    return 0.0
                return round(len(wa & wb) / max(len(wa | wb), 1), 4)

            result[f"{c1}_{c2}_jaccard"] = [jaccard(a, b) for a, b in zip(s1, s2)]
            avg = result[f"{c1}_{c2}_jaccard"].mean()
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Jaccard similarity between '{c1}' and '{c2}': avg={avg:.4f}",
            )
        else:
            # Consecutive row overlap within single column
            text_cols = _get_text_cols(df, col)
            if not text_cols:
                return HandlerResult(success=False, error="No text columns found. Provide columns=[col1, col2] or column=col")
            target = text_cols[0]
            s = result[target].fillna("").astype(str).str.lower()
            overlaps: list[float] = [0.0]
            for i in range(1, len(s)):
                wa = set(re.findall(r"\b\w+\b", s.iloc[i - 1]))
                wb = set(re.findall(r"\b\w+\b", s.iloc[i]))
                j = len(wa & wb) / max(len(wa | wb), 1) if (wa or wb) else 0.0
                overlaps.append(round(j, 4))
            result[f"{target}_row_overlap"] = overlaps
            return HandlerResult(
                success=True, result_df=result, output_type="generate",
                summary=f"Row-to-row Jaccard overlap for '{target}': avg={np.mean(overlaps):.4f}",
            )

    # ── 38. Truncate or pad text to fixed length ─────────────────────────

    @staticmethod
    def handle_text_truncate_pad(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Truncate or pad text to a fixed word count.
        Useful for preparing uniform-length input for models."""
        col = params.get("column")
        max_words = int(params.get("max_words", 128))
        pad_token = params.get("pad_token", "")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        for c in text_cols:
            def trunc_pad(text: str) -> str:
                words = text.split()[:max_words]
                if pad_token and len(words) < max_words:
                    words.extend([pad_token] * (max_words - len(words)))
                return " ".join(words)

            result[c] = result[c].fillna("").astype(str).apply(trunc_pad)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Truncated/padded {len(text_cols)} column(s) to {max_words} words",
        )

    # ── 39. Text length distribution analysis ────────────────────────────

    @staticmethod
    def handle_text_length_dist(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Analyze text length distribution (char & word count) with charts.
        Useful for choosing chunk size, max_len, or detecting anomalies."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        s = df[target].fillna("").astype(str)
        char_lens = s.str.len()
        word_lens = s.str.split().str.len().fillna(0).astype(int)

        stats = pd.DataFrame({
            "metric": ["count", "mean_chars", "median_chars", "std_chars", "min_chars", "max_chars",
                        "mean_words", "median_words", "min_words", "max_words",
                        "empty_rows", "single_word_rows"],
            "value": [len(s), round(float(char_lens.mean()), 1), int(char_lens.median()),
                      round(float(char_lens.std()), 1), int(char_lens.min()), int(char_lens.max()),
                      round(float(word_lens.mean()), 1), int(word_lens.median()),
                      int(word_lens.min()), int(word_lens.max()),
                      int((char_lens == 0).sum()), int((word_lens <= 1).sum())],
        })

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go_fig
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Character Length", "Word Count"])
        fig.add_trace(go_fig.Histogram(x=char_lens, nbinsx=30, marker_color="#FB8C3C", name="Chars"), row=1, col=1)
        fig.add_trace(go_fig.Histogram(x=word_lens, nbinsx=30, marker_color="#2EC4B6", name="Words"), row=1, col=2)
        _style(fig, title=f"Text Length Distribution — {target} (n={len(s)})")
        fig.update_layout(showlegend=False, height=350)

        return HandlerResult(
            success=True, result_df=stats, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=f"'{target}': avg {word_lens.mean():.0f} words/doc, range [{word_lens.min()}-{word_lens.max()}], {(char_lens==0).sum()} empty",
        )

    # ── 40. Extract unique / rare words per document ─────────────────────

    @staticmethod
    def handle_text_unique_words(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Extract words that appear only in one document (unique to that doc).
        Creates a column with comma-separated rare/unique words per row."""
        col = params.get("column")
        result = df.copy()
        text_cols = _get_text_cols(result, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        corpus = result[target].fillna("").astype(str).str.lower()

        # Build document frequency
        doc_freq: Counter = Counter()
        doc_words: list[set] = []
        for text in corpus:
            words = set(re.findall(r"\b\w+\b", text)) - ENGLISH_STOPWORDS
            doc_words.append(words)
            doc_freq.update(words)

        # Words appearing in only 1 document
        unique_to_doc: list[str] = []
        for words in doc_words:
            uniques = sorted(w for w in words if doc_freq[w] == 1)
            unique_to_doc.append(", ".join(uniques[:10]))

        result[f"{target}_unique_words"] = unique_to_doc
        result[f"{target}_unique_count"] = [len(w.split(", ")) if w else 0 for w in unique_to_doc]

        total_unique = sum(1 for _, cnt in doc_freq.items() if cnt == 1)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Found {total_unique} corpus-unique words across {len(corpus)} documents",
        )

    # ── 41. Exact text deduplication (fast) ──────────────────────────────

    @staticmethod
    def handle_text_dedup_exact(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Remove rows with exact duplicate text (case-insensitive).
        Much faster than similarity-based dedup for large datasets."""
        col = params.get("column")
        keep = params.get("keep", "first")  # first | last
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        result = df.copy()
        result["_text_lower"] = result[target].fillna("").astype(str).str.lower().str.strip()
        original = len(result)
        result = result.drop_duplicates(subset="_text_lower", keep=keep).drop(columns="_text_lower").reset_index(drop=True)
        removed = original - len(result)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Exact dedup on '{target}': {original} → {len(result)} rows (removed {removed} duplicates)",
        )

    # ── 42. Split text by paragraphs ─────────────────────────────────────

    @staticmethod
    def handle_text_to_paragraphs(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Split text into paragraphs (by blank lines). Each paragraph becomes a new row."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        rows: list[dict] = []

        for _, row in df.iterrows():
            text = str(row.get(target, ""))
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            for i, para in enumerate(paragraphs):
                new_row = row.to_dict()
                new_row[target] = para
                new_row["_para_id"] = i
                new_row["_para_total"] = len(paragraphs)
                rows.append(new_row)

        result = pd.DataFrame(rows).reset_index(drop=True)
        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Split '{target}': {len(df)} docs → {len(result)} paragraphs",
        )

    # ── 43. Count pattern occurrences ────────────────────────────────────

    @staticmethod
    def handle_text_count_pattern(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Count occurrences of a specific pattern (word/phrase/regex) per row.
        Creates a count column and optionally filters rows with matches."""
        col = params.get("column")
        pattern = params.get("pattern", "")
        filter_matches = params.get("filter", False)
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")
        if not pattern:
            return HandlerResult(success=False, error="Specify pattern= parameter")

        target = text_cols[0]
        result = df.copy()
        s = result[target].fillna("").astype(str)
        try:
            result[f"count_{pattern[:20]}"] = s.str.count(pattern)
        except re.error:
            result[f"count_{pattern[:20]}"] = s.str.count(re.escape(pattern))

        total = int(result[f"count_{pattern[:20]}"].sum())
        rows_with = int((result[f"count_{pattern[:20]}"] > 0).sum())

        if filter_matches:
            result = result[result[f"count_{pattern[:20]}"] > 0].reset_index(drop=True)

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Pattern '{pattern}': {total} occurrences in {rows_with}/{len(df)} rows",
        )

    # ── 44. Text dataset summary report ──────────────────────────────────

    @staticmethod
    def handle_text_summary_report(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Comprehensive text dataset report: language mix, length stats,
        vocabulary, quality issues, and recommendations."""
        col = params.get("column")
        text_cols = _get_text_cols(df, col)
        if not text_cols:
            return HandlerResult(success=False, error="No text columns found")

        target = text_cols[0]
        s = df[target].fillna("").astype(str)
        findings: list[dict] = []

        # Basic stats
        empty = int((s.str.strip() == "").sum())
        word_counts = s.str.split().str.len().fillna(0)
        char_counts = s.str.len()
        findings.append({"category": "Size", "detail": f"{len(s)} documents, {int(word_counts.sum()):,} total words"})
        findings.append({"category": "Length", "detail": f"avg {word_counts.mean():.0f} words/doc, range [{int(word_counts.min())}-{int(word_counts.max())}]"})

        if empty > 0:
            findings.append({"category": "Quality", "detail": f"{empty} empty rows ({empty/len(s)*100:.1f}%)"})

        # Vocabulary
        all_words: list[str] = []
        for text in s.str.lower():
            all_words.extend(re.findall(r"\b\w+\b", text))
        vocab_size = len(set(all_words))
        findings.append({"category": "Vocabulary", "detail": f"{vocab_size:,} unique words, TTR={vocab_size/max(len(all_words),1):.4f}"})

        # Duplicates
        dup_count = s.str.lower().str.strip().duplicated().sum()
        if dup_count > 0:
            findings.append({"category": "Quality", "detail": f"{int(dup_count)} exact duplicates ({dup_count/len(s)*100:.1f}%)"})

        # Language detection sample
        def detect_script(text: str) -> str:
            for ch in text[:100]:
                cp = ord(ch)
                if 0x0E00 <= cp <= 0x0E7F: return "Thai"
                if 0x4E00 <= cp <= 0x9FFF: return "Chinese"
                if 0x3040 <= cp <= 0x30FF: return "Japanese"
            return "Latin"

        scripts = s.head(100).apply(detect_script).value_counts()
        findings.append({"category": "Language", "detail": ", ".join(f"{k}: {v}" for k, v in scripts.items())})

        # Short texts
        short = int((word_counts < 5).sum())
        if short > 0:
            findings.append({"category": "Quality", "detail": f"{short} very short texts (<5 words)"})

        # Recommendations
        recs: list[str] = []
        if empty > 0:
            recs.append(f"Remove {empty} empty rows with nlp.text_filter")
        if dup_count > 0:
            recs.append(f"Remove {int(dup_count)} duplicates with nlp.text_dedup_exact")
        if word_counts.max() > 500:
            recs.append(f"Consider nlp.text_chunk for long documents (max={int(word_counts.max())} words)")
        if vocab_size > 10000:
            recs.append("Large vocabulary — consider nlp.text_normalize or nlp.remove_stopwords")

        for rec in recs:
            findings.append({"category": "Recommendation", "detail": rec})

        result_df = pd.DataFrame(findings)

        # Chart
        fig = px.histogram(x=word_counts, nbins=30)
        fig.update_traces(marker_color="#FB8C3C")
        _style(fig, title=f"Text Report — {target}: {len(s)} docs, {vocab_size:,} vocab")
        fig.update_layout(xaxis_title="Words per document", yaxis_title="Count")

        summary = "\n".join(f"• [{r['category']}] {r['detail']}" for r in findings)
        return HandlerResult(
            success=True, result_df=result_df, output_type="query",
            charts_plotly=[fig.to_json()],
            summary=summary,
        )

    # ── 45. Text stratified sample ───────────────────────────────────────

    @staticmethod
    def handle_text_stratified_sample(df: pd.DataFrame, params: dict) -> HandlerResult:
        """Take a stratified random sample maintaining label distribution.
        Useful for creating balanced train/test splits or review subsets."""
        label_col = params.get("column")
        n = int(params.get("n", 100))
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not label_col or label_col not in df.columns:
            candidates = [c for c in cat_cols if df[c].nunique() <= 30]
            label_col = candidates[0] if candidates else None
        if label_col is None:
            return HandlerResult(success=False, error="No label column for stratified sampling")

        counts = df[label_col].value_counts()
        n_classes = len(counts)
        per_class = max(1, n // n_classes)

        parts: list[pd.DataFrame] = []
        for label in counts.index:
            subset = df[df[label_col] == label]
            sample_n = min(per_class, len(subset))
            parts.append(subset.sample(n=sample_n, random_state=42))

        result = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        new_dist = result[label_col].value_counts().to_dict()

        return HandlerResult(
            success=True, result_df=result, output_type="generate",
            summary=f"Stratified sample: {len(df)} → {len(result)} rows, {n_classes} classes. Distribution: {new_dist}",
        )

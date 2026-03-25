"""NLP / Text preprocessing handler — 15 handlers for text cleaning,
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

"""Shared NLP constants and helpers."""
from __future__ import annotations

import pandas as pd

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

_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    "url": r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    "hashtag": r"#\w+",
    "mention": r"@\w+",
    "phone": r"\+?\d[\d\-\s()]{7,}\d",
    "number": r"-?\d+\.?\d*",
}

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


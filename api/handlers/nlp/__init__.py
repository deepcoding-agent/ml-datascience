"""nlp handlers — individual modules."""
from api.handlers.nlp.text_clean import handle_text_clean
from api.handlers.nlp.remove_stopwords import handle_remove_stopwords
from api.handlers.nlp.tokenize import handle_tokenize
from api.handlers.nlp.tfidf import handle_tfidf
from api.handlers.nlp.bow import handle_bow
from api.handlers.nlp.ngrams import handle_ngrams
from api.handlers.nlp.regex_extract import handle_regex_extract
from api.handlers.nlp.sentiment_score import handle_sentiment_score
from api.handlers.nlp.word_frequency import handle_word_frequency
from api.handlers.nlp.text_similarity import handle_text_similarity
from api.handlers.nlp.vocab_stats import handle_vocab_stats
from api.handlers.nlp.text_normalize import handle_text_normalize
from api.handlers.nlp.language_detect import handle_language_detect
from api.handlers.nlp.hash_vectorize import handle_hash_vectorize
from api.handlers.nlp.text_encode import handle_text_encode
from api.handlers.nlp.keyword_extract import handle_keyword_extract
from api.handlers.nlp.char_features import handle_char_features
from api.handlers.nlp.sentence_features import handle_sentence_features
from api.handlers.nlp.readability_score import handle_readability_score
from api.handlers.nlp.text_dedup import handle_text_dedup
from api.handlers.nlp.emoji_features import handle_emoji_features
from api.handlers.nlp.text_mask_pii import handle_text_mask_pii
from api.handlers.nlp.text_augment import handle_text_augment
from api.handlers.nlp.collocations import handle_collocations
from api.handlers.nlp.word_cloud import handle_word_cloud
from api.handlers.nlp.text_filter import handle_text_filter
from api.handlers.nlp.class_balance_text import handle_class_balance_text
from api.handlers.nlp.text_chunk import handle_text_chunk
from api.handlers.nlp.spelling_features import handle_spelling_features
from api.handlers.nlp.text_concat import handle_text_concat
from api.handlers.nlp.text_replace import handle_text_replace
from api.handlers.nlp.text_split_sentences import handle_text_split_sentences
from api.handlers.nlp.text_oversample import handle_text_oversample
from api.handlers.nlp.doc_term_matrix import handle_doc_term_matrix
from api.handlers.nlp.text_window import handle_text_window
from api.handlers.nlp.text_label_rules import handle_text_label_rules
from api.handlers.nlp.word_overlap import handle_word_overlap
from api.handlers.nlp.text_truncate_pad import handle_text_truncate_pad
from api.handlers.nlp.text_length_dist import handle_text_length_dist
from api.handlers.nlp.text_unique_words import handle_text_unique_words
from api.handlers.nlp.text_dedup_exact import handle_text_dedup_exact
from api.handlers.nlp.text_to_paragraphs import handle_text_to_paragraphs
from api.handlers.nlp.text_count_pattern import handle_text_count_pattern
from api.handlers.nlp.text_summary_report import handle_text_summary_report
from api.handlers.nlp.text_stratified_sample import handle_text_stratified_sample
from api.handlers.nlp.text_ngram_frequency import handle_text_ngram_frequency
from api.handlers.nlp.text_extract_numbers import handle_text_extract_numbers
from api.handlers.nlp.text_remove_rare import handle_text_remove_rare
from api.handlers.nlp.text_pos_patterns import handle_text_pos_patterns
from api.handlers.nlp.text_diversity_index import handle_text_diversity_index

__all__ = [
    "handle_text_clean",
    "handle_remove_stopwords",
    "handle_tokenize",
    "handle_tfidf",
    "handle_bow",
    "handle_ngrams",
    "handle_regex_extract",
    "handle_sentiment_score",
    "handle_word_frequency",
    "handle_text_similarity",
    "handle_vocab_stats",
    "handle_text_normalize",
    "handle_language_detect",
    "handle_hash_vectorize",
    "handle_text_encode",
    "handle_keyword_extract",
    "handle_char_features",
    "handle_sentence_features",
    "handle_readability_score",
    "handle_text_dedup",
    "handle_emoji_features",
    "handle_text_mask_pii",
    "handle_text_augment",
    "handle_collocations",
    "handle_word_cloud",
    "handle_text_filter",
    "handle_class_balance_text",
    "handle_text_chunk",
    "handle_spelling_features",
    "handle_text_concat",
    "handle_text_replace",
    "handle_text_split_sentences",
    "handle_text_oversample",
    "handle_doc_term_matrix",
    "handle_text_window",
    "handle_text_label_rules",
    "handle_word_overlap",
    "handle_text_truncate_pad",
    "handle_text_length_dist",
    "handle_text_unique_words",
    "handle_text_dedup_exact",
    "handle_text_to_paragraphs",
    "handle_text_count_pattern",
    "handle_text_summary_report",
    "handle_text_stratified_sample",
    "handle_text_ngram_frequency",
    "handle_text_extract_numbers",
    "handle_text_remove_rare",
    "handle_text_pos_patterns",
    "handle_text_diversity_index",
]

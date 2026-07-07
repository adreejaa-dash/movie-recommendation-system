"""
feature_engineering.py
Build weighted tag strings and vectorize with TF-IDF / CountVectorizer.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────────────────
# Simple English stopwords (avoids NLTK download requirement)
# ──────────────────────────────────────────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "he", "she", "they", "we", "you", "i", "me", "him", "her", "us", "them",
    "his", "their", "our", "your", "my", "its", "this", "that", "these",
    "those", "which", "who", "whom", "what", "where", "when", "why", "how",
    "as", "if", "into", "through", "during", "including", "until", "while",
    "of", "about", "against", "between", "out", "up", "then", "than",
    "s", "t", "just", "because", "as", "until", "while", "although",
    "after", "before", "about", "over", "each", "all", "more", "also"
}


def _clean_token(text: str) -> str:
    """Lowercase, remove punctuation/numbers."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.strip()


def _tokenize_and_clean(text: str, remove_stops: bool = True) -> list[str]:
    tokens = _clean_token(text).split()
    if remove_stops:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    return tokens


def _collapse(items: list[str]) -> str:
    """Join list items into a single no-space token (for multi-word names)."""
    return " ".join(i.replace(" ", "") for i in items)


# ──────────────────────────────────────────────────────────────────────────────
# Tag builder — weighted repetition boosts importance
# ──────────────────────────────────────────────────────────────────────────────

def build_tags(row: pd.Series) -> str:
    """
    Combine movie features into a single tag string.
    Weighting strategy (via repetition):
      director  × 3  (very strong signal)
      genres    × 2
      cast      × 2  (top 5 actors)
      keywords  × 1
      overview  × 1  (cleaned, stopwords removed)
    """
    genres   = _collapse(row.get("genres", []) or [])
    keywords = _collapse(row.get("keywords", []) or [])
    cast     = _collapse(row.get("cast", []) or [])
    director = (row.get("director") or "").replace(" ", "")
    overview_tokens = _tokenize_and_clean(row.get("overview", "") or "")
    overview = " ".join(overview_tokens)

    parts = (
        [director] * 3         # director weighted ×3
        + genres.split() * 2   # genres weighted ×2
        + cast.split() * 2     # cast weighted ×2
        + keywords.split()     # keywords ×1
        + overview.split()     # overview ×1
    )
    return " ".join(parts)


def add_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'tags' column to the dataframe."""
    df = df.copy()
    df["tags"] = df.apply(build_tags, axis=1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Vectorization
# ──────────────────────────────────────────────────────────────────────────────

def vectorize(
    df: pd.DataFrame,
    method: str = "tfidf",
    max_features: int = 10_000,
) -> tuple:
    """
    Vectorize the 'tags' column.

    Args:
        method: 'tfidf' or 'count'
        max_features: vocabulary size cap

    Returns:
        (matrix, vectorizer, similarity_matrix)
    """
    if method == "tfidf":
        vec = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )
    else:
        vec = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
        )

    matrix = vec.fit_transform(df["tags"])
    sim = cosine_similarity(matrix, matrix)
    return matrix, vec, sim


if __name__ == "__main__":
    from data_preprocessing import load_and_clean
    df = load_and_clean()
    df = add_tags(df)
    print("Sample tags:", df["tags"].iloc[0][:200])
    _, _, sim = vectorize(df, method="tfidf")
    print("Similarity matrix shape:", sim.shape)

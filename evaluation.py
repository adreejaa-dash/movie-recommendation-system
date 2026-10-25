"""
evaluation.py
Quick evaluation of content-based recommenders:
  - Qualitative TF-IDF evaluation
  - Precision@k (using genre overlap as a proxy for relevance)
"""

import pandas as pd
from data_preprocessing import load_and_clean
from recommender import build_content_recommender


def precision_at_k(recommender, query_title: str, k: int = 10) -> float:
    """
    Proxy metric: what fraction of top-k recommendations share
    at least one genre with the query movie?
    """
    idx = recommender._idx(query_title)
    if idx is None:
        return 0.0
    query_genres = set(recommender.df.iloc[idx]["genres"])
    recs = recommender.recommend(query_title, top_n=k)
    if recs.empty:
        return 0.0
    hits = sum(
        1 for genres in recs["genres"]
        if set(genres) & query_genres
    )
    return hits / k


def evaluate_recommender(query_movies: list[str], k: int = 10):
    """Print an evaluation table for TF-IDF Precision@k."""
    df = load_and_clean()
    rec = build_content_recommender(df)

    print(f"\n{'Movie':<35} {'P@k TF-IDF':>12}")
    print("-" * 49)
    for movie in query_movies:
        p_tfidf = precision_at_k(rec, movie, k)
        print(f"{movie:<35} {p_tfidf:>12.2f}")
    print()

    # Qualitative: show top-5 recs for first movie
    m = query_movies[0]
    print(f"\n=== Top-5 recommendations for '{m}' (TF-IDF) ===\n")
    t = rec.recommend(m, top_n=5)
    if not t.empty:
        for _, r in t.iterrows():
            print(f"  {r['title']:<35} sim={r['similarity']:.3f}")


# Backward-compatible alias
compare_methods = evaluate_recommender


if __name__ == "__main__":
    SAMPLE_MOVIES = [
        "The Dark Knight",
        "Inception",
        "The Avengers",
        "Interstellar",
        "The Godfather",
    ]
    evaluate_recommender(SAMPLE_MOVIES)

"""
evaluation.py
Quick evaluation of content-based recommenders:
  - Qualitative TF-IDF vs CountVectorizer comparison
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


def compare_methods(query_movies: list[str], k: int = 10):
    """Print a side-by-side comparison table for TF-IDF vs CountVec."""
    df = load_and_clean()
    rec_tfidf = build_content_recommender(df, method="tfidf")
    rec_count = build_content_recommender(df, method="count")

    print(f"\n{'Movie':<35} {'P@k TF-IDF':>12} {'P@k Count':>12}")
    print("-" * 62)
    for movie in query_movies:
        p_tfidf = precision_at_k(rec_tfidf, movie, k)
        p_count = precision_at_k(rec_count, movie, k)
        print(f"{movie:<35} {p_tfidf:>12.2f} {p_count:>12.2f}")
    print()

    # Qualitative: show top-5 recs for each method for first movie
    m = query_movies[0]
    print(f"\n=== Top-5 recommendations for '{m}' ===\n")
    print("── TF-IDF ──")
    t = rec_tfidf.recommend(m, top_n=5)
    if not t.empty:
        for _, r in t.iterrows():
            print(f"  {r['title']:<35} sim={r['similarity']:.3f}")
    print("\n── CountVectorizer ──")
    c = rec_count.recommend(m, top_n=5)
    if not c.empty:
        for _, r in c.iterrows():
            print(f"  {r['title']:<35} sim={r['similarity']:.3f}")


if __name__ == "__main__":
    SAMPLE_MOVIES = [
        "The Dark Knight",
        "Inception",
        "The Avengers",
        "Interstellar",
        "The Godfather",
    ]
    compare_methods(SAMPLE_MOVIES)

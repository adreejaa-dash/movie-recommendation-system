"""
recommender.py
Content-based recommender + optional collaborative filtering layer.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from feature_engineering import add_tags, vectorize


# ──────────────────────────────────────────────────────────────────────────────
# Content-Based Recommender
# ──────────────────────────────────────────────────────────────────────────────

class ContentRecommender:
    """
    TF-IDF / CountVec cosine-similarity recommender with explainability.
    """

    def __init__(self, df: pd.DataFrame, method: str = "tfidf"):
        self.df = df.reset_index(drop=True)
        self.method = method
        self._build(method)

    def _build(self, method: str):
        self.matrix, self.vectorizer, self.sim = vectorize(self.df, method=method)
        # Index title → row index for fast lookup
        self.title_to_idx = {
            title.lower(): i for i, title in enumerate(self.df["title"])
        }

    def _idx(self, title: str) -> int | None:
        return self.title_to_idx.get(title.lower())

    def recommend(
        self,
        title: str,
        top_n: int = 10,
        user_profile: dict | None = None,
    ) -> pd.DataFrame:
        """
        Return top_n recommendations for a given movie title.

        Args:
            title: exact movie title (case-insensitive)
            top_n: number of results
            user_profile: optional dict {genre: weight, actor: weight, ...}
                          used to re-rank results

        Returns:
            DataFrame with columns: title, similarity, genres, cast, director,
                                    vote_average, why
        """
        idx = self._idx(title)
        if idx is None:
            return pd.DataFrame()

        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Exclude the query movie itself
        scores = [(i, s) for i, s in scores if i != idx][:top_n]

        query_row = self.df.iloc[idx]
        results = []
        for rank, (i, score) in enumerate(scores):
            row = self.df.iloc[i]
            why = self._explain(query_row, row)

            # Optional: boost score by user profile match
            final_score = score
            if user_profile:
                boost = self._profile_boost(row, user_profile)
                final_score = min(1.0, score + boost)

            results.append({
                "title": row["title"],
                "similarity": round(final_score, 4),
                "genres": row["genres"],
                "cast": row["cast"],
                "director": row["director"],
                "vote_average": row["vote_average"],
                "why": why,
                "movie_id": row.get("movie_id", i),
            })

        return pd.DataFrame(results)

    def _explain(self, query: pd.Series, candidate: pd.Series) -> dict:
        """Return overlapping features between query and candidate."""
        shared_genres   = list(set(query["genres"])   & set(candidate["genres"]))
        shared_cast     = list(set(query["cast"])     & set(candidate["cast"]))
        shared_director = (
            query["director"] if query["director"] == candidate["director"]
            else None
        )
        shared_keywords = list(
            set(query.get("keywords", []) or [])
            & set(candidate.get("keywords", []) or [])
        )[:5]

        return {
            "shared_genres":   shared_genres,
            "shared_cast":     shared_cast,
            "shared_director": shared_director,
            "shared_keywords": shared_keywords,
        }

    def _profile_boost(self, row: pd.Series, profile: dict) -> float:
        """Compute a small score boost based on genre/actor preference."""
        boost = 0.0
        genres = row.get("genres", []) or []
        cast   = row.get("cast", []) or []
        for item in genres:
            boost += profile.get(item, 0.0) * 0.05
        for actor in cast:
            boost += profile.get(actor, 0.0) * 0.03
        director = row.get("director", "")
        boost += profile.get(director, 0.0) * 0.04
        return min(boost, 0.2)   # cap boost at 0.2

    def search_titles(self, query: str, top_n: int = 10) -> list[str]:
        """Fuzzy search for movie titles (simple substring match)."""
        q = query.lower()
        matches = [t for t in self.df["title"] if q in t.lower()]
        return matches[:top_n]


# ──────────────────────────────────────────────────────────────────────────────
# Collaborative Filtering (item-based, cosine sim on ratings matrix)
# ──────────────────────────────────────────────────────────────────────────────

class CollaborativeRecommender:
    """
    Simple item-based collaborative filtering using a ratings matrix.
    Works with MovieLens data or a simulated ratings matrix.
    """

    def __init__(self, ratings_df: pd.DataFrame, title_map: dict | None = None):
        """
        Args:
            ratings_df: DataFrame with columns [userId, movieId, rating]
            title_map:  dict {movieId: title} — optional display names
        """
        self.title_map = title_map or {}
        self._build(ratings_df)

    def _build(self, ratings_df: pd.DataFrame):
        # Pivot to user×movie matrix, fill NaN with 0
        self.matrix = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating"
        ).fillna(0)
        # Item-based: transpose so each row = movie, each col = user
        self.item_matrix = self.matrix.T
        self.sim = cosine_similarity(self.item_matrix)
        self.sim_df = pd.DataFrame(
            self.sim,
            index=self.item_matrix.index,
            columns=self.item_matrix.index,
        )
        self.movie_ids = list(self.item_matrix.index)

    def recommend(self, movie_id, top_n: int = 10) -> pd.DataFrame:
        """Return top_n collaboratively similar movies."""
        if movie_id not in self.sim_df.index:
            return pd.DataFrame()
        scores = self.sim_df[movie_id].drop(movie_id).sort_values(ascending=False)
        top = scores.head(top_n)
        return pd.DataFrame({
            "movie_id": top.index,
            "title": [self.title_map.get(mid, str(mid)) for mid in top.index],
            "cf_similarity": top.values.round(4),
        })


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_content_recommender(df: pd.DataFrame, method: str = "tfidf") -> ContentRecommender:
    """Add tags to df and return a ready ContentRecommender."""
    df = add_tags(df)
    return ContentRecommender(df, method=method)


def simulate_collab_recommender(df: pd.DataFrame, n_users: int = 200) -> CollaborativeRecommender:
    """
    Simulate a ratings matrix from TMDB data when MovieLens is unavailable.
    Users prefer movies in genres they randomly favour.
    """
    rng = np.random.default_rng(42)
    all_genres = list({g for genres in df["genres"] for g in genres})
    records = []
    for uid in range(n_users):
        fav_genres = rng.choice(all_genres, size=rng.integers(2, 5), replace=False)
        # Find movies matching those genres
        mask = df["genres"].apply(lambda gs: any(g in gs for g in fav_genres))
        pool = df[mask]
        n_rate = min(len(pool), rng.integers(10, 40))
        sampled = pool.sample(n=n_rate, random_state=uid)
        for _, row in sampled.iterrows():
            genre_overlap = sum(1 for g in row["genres"] if g in fav_genres)
            base_rating = 3.0 + genre_overlap * 0.4 + rng.normal(0, 0.5)
            rating = float(np.clip(base_rating, 1.0, 5.0))
            records.append({"userId": uid, "movieId": row["movie_id"], "rating": rating})

    ratings_df = pd.DataFrame(records)
    title_map = dict(zip(df["movie_id"], df["title"]))
    return CollaborativeRecommender(ratings_df, title_map=title_map)


if __name__ == "__main__":
    from data_preprocessing import load_and_clean
    df = load_and_clean()
    rec = build_content_recommender(df)
    results = rec.recommend("The Dark Knight", top_n=5)
    if not results.empty:
        for _, r in results.iterrows():
            print(f"  {r['title']} ({r['similarity']:.3f}) — {r['why']}")
    else:
        print("Movie not found. Check title spelling.")

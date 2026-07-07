"""
ratings_store.py
Persist and retrieve per-user movie ratings in a local SQLite database.
Also computes a simple user preference profile for re-ranking.
"""

import sqlite3
import os
from datetime import datetime
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "ratings.db")


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            user_id   TEXT    NOT NULL,
            movie_id  INTEGER NOT NULL,
            title     TEXT,
            rating    REAL    NOT NULL,
            rated_at  TEXT    NOT NULL,
            PRIMARY KEY (user_id, movie_id)
        )
    """)
    conn.commit()
    return conn


def save_rating(movie_id: int, title: str, rating: float, user_id: str = "default"):
    """Insert or replace a user's rating for a movie."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO ratings
           (user_id, movie_id, title, rating, rated_at)
           VALUES (?, ?, ?, ?, ?)""",
        (user_id, movie_id, title, rating, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_ratings(user_id: str = "default") -> pd.DataFrame:
    """Return all ratings for a user as a DataFrame."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM ratings WHERE user_id = ? ORDER BY rated_at DESC",
        conn, params=(user_id,)
    )
    conn.close()
    return df


def build_user_profile(user_id: str = "default", movies_df: pd.DataFrame = None) -> dict:
    """
    Build a preference profile dict {genre/actor/director: weight} from ratings.
    Only considers movies rated >= 3.5 (positive signal).
    Returns empty dict if no ratings found.
    """
    ratings = get_ratings(user_id)
    if ratings.empty or movies_df is None:
        return {}

    liked = ratings[ratings["rating"] >= 3.5]
    if liked.empty:
        return {}

    profile: dict[str, float] = {}
    for _, r in liked.iterrows():
        movie_row = movies_df[movies_df["movie_id"] == r["movie_id"]]
        if movie_row.empty:
            continue
        row = movie_row.iloc[0]
        weight = (r["rating"] - 2.5) / 2.5   # 0.4 for 3.5★, 1.0 for 5★

        for genre in (row.get("genres") or []):
            profile[genre] = profile.get(genre, 0.0) + weight
        for actor in (row.get("cast") or []):
            profile[actor] = profile.get(actor, 0.0) + weight * 0.7
        director = row.get("director") or ""
        if director:
            profile[director] = profile.get(director, 0.0) + weight * 0.9

    # Normalize
    if profile:
        max_val = max(profile.values())
        profile = {k: round(v / max_val, 4) for k, v in profile.items()}

    return profile

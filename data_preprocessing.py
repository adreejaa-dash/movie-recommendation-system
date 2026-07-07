"""
data_preprocessing.py
Load, merge, and clean the TMDB 5000 Movie Dataset.
Expected files in ./data/: tmdb_5000_movies.csv, tmdb_5000_credits.csv
"""

import ast
import re
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to parse JSON-like string columns
# ──────────────────────────────────────────────────────────────────────────────

def _safe_eval(val):
    """Convert stringified list/dict to Python object; return [] on failure."""
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []


def _extract_names(val, key="name", top_n=None):
    """Pull 'key' from each dict in a list; optionally limit to top_n."""
    items = _safe_eval(val) if isinstance(val, str) else val
    names = [d[key] for d in items if isinstance(d, dict) and key in d]
    return names[:top_n] if top_n else names


def _extract_director(crew_val):
    """Return director name from crew list or empty string."""
    crew = _safe_eval(crew_val) if isinstance(crew_val, str) else crew_val
    for member in crew:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Main loader
# ──────────────────────────────────────────────────────────────────────────────

def load_and_clean(
    movies_path: str = "data/tmdb_5000_movies.csv",
    credits_path: str = "data/tmdb_5000_credits.csv",
) -> pd.DataFrame:
    """
    Load TMDB CSVs, merge on movie_id, extract clean feature columns.

    Returns a DataFrame with columns:
        movie_id, title, overview, genres, keywords, cast, director,
        vote_average, vote_count, popularity
    """
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)

    # Normalise id column name
    if "movie_id" not in credits.columns and "id" in credits.columns:
        credits = credits.rename(columns={"id": "movie_id"})
    if "movie_id" not in movies.columns and "id" in movies.columns:
        movies = movies.rename(columns={"id": "movie_id"})

    # Drop duplicate title from credits before merging to avoid _x/_y collision
    credits = credits.drop(columns=[c for c in ["title"] if c in credits.columns])

    df = movies.merge(credits, on="movie_id", how="inner")

    # Drop duplicates & rows with no title
    df.drop_duplicates(subset="movie_id", inplace=True)
    df.dropna(subset=["title"], inplace=True)

    # Parse structured columns
    df["genres"]   = df["genres"].apply(lambda x: _extract_names(x))
    df["keywords"] = df["keywords"].apply(lambda x: _extract_names(x))
    df["cast"]     = df["cast"].apply(lambda x: _extract_names(x, top_n=5))
    df["director"] = df["crew"].apply(_extract_director)

    # Clean overview
    df["overview"] = df["overview"].fillna("").astype(str)

    # Keep useful numeric cols
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0)
    df["vote_count"]   = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df["popularity"]   = pd.to_numeric(df.get("popularity", 0), errors="coerce").fillna(0)

    keep = ["movie_id", "title", "overview", "genres", "keywords",
            "cast", "director", "vote_average", "vote_count", "popularity"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)

    return df


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Loaded {len(df)} movies")
    print(df[["title", "genres", "cast", "director"]].head(3))

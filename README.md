# 🎬 CineMatch — AI Movie Recommendation System

> A portfolio-grade content-based + collaborative filtering recommender with a Streamlit UI, explainable recommendations, and user rating personalization.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![SQLite](https://img.shields.io/badge/SQLite-local%20ratings-green)

---

## Problem Statement

Recommending movies users will actually enjoy is a core challenge in information retrieval. This project implements a full end-to-end recommendation pipeline — from raw CSV data to an interactive web app — using two complementary approaches:

- **Content-Based Filtering** — recommend movies similar to one you like (based on genres, cast, director, plot keywords)
- **Collaborative Filtering** — recommend movies liked by users with similar taste

---

## Dataset

| Dataset | Source | Files | Size | Status |
|---------|--------|-------|------|--------|
| TMDB 5000 Movies | [Kaggle ↗](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | `tmdb_5000_movies.csv` | ~5 MB | ✅ included in repo |
| TMDB 5000 Credits | [Kaggle ↗](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | `tmdb_5000_credits.csv` | ~38 MB | ⚠️ download manually (see setup) |
| MovieLens (optional) | [GroupLens ↗](https://grouplens.org/datasets/movielens/) | `ratings.csv` | varies | ℹ️ simulated if absent |

> **Note:** `tmdb_5000_credits.csv` (~38 MB) is excluded from this repo via `.gitignore`. Download it from the Kaggle link above and place it in `data/` before running the app.

---

## Screenshot

> _Run `streamlit run app.py`, then add a screenshot here._
>
> The app features a cinematic dark theme with a hero header, movie search/dropdown, content-based recommendations with similarity scores and "why recommended" chips (shared director/genre/cast), collaborative filtering panel, and a star rating widget per movie.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CineMatch Pipeline                      │
│                                                         │
│  TMDB CSVs ──► data_preprocessing.py                   │
│                     │                                   │
│                     ▼                                   │
│              feature_engineering.py                     │
│          (weighted tags: dir×3, genre×2, cast×2)        │
│                     │                                   │
│            ┌────────┴────────┐                          │
│            ▼                 ▼                          │
│        TF-IDF           CountVectorizer                 │
│            └────────┬────────┘                          │
│                     ▼                                   │
│              recommender.py                             │
│         ┌───────────┴───────────┐                       │
│         ▼                       ▼                       │
│  ContentRecommender    CollaborativeRecommender          │
│   (cosine sim +         (item-based CF,                 │
│    explainability)       ratings matrix)                │
│         │                       │                       │
│         └───────────┬───────────┘                       │
│                     ▼                                   │
│              ratings_store.py                           │
│         (SQLite · user profiles · re-ranking)           │
│                     │                                   │
│                     ▼                                   │
│                  app.py  (Streamlit UI)                 │
└─────────────────────────────────────────────────────────┘
```

---

## Feature Engineering

The key insight: instead of feeding raw text to the vectorizer, we build a **weighted tag string** per movie:

```
tags = [director] × 3  +  [genres] × 2  +  [cast] × 2  +  [keywords]  +  [overview_tokens]
```

This biases cosine similarity toward shared director/genre/cast — the strongest relevance signals — without requiring a custom kernel.

**Example result** for *The Dark Knight*:

| # | Recommended Movie | Similarity | Why |
|---|-------------------|-----------|-----|
| 1 | The Dark Knight Rises | 0.499 | Dir: Christopher Nolan, Genre: Crime, Genre: Drama |
| 2 | Batman Begins | 0.455 | Dir: Christopher Nolan, Genre: Crime, Genre: Drama |
| 3 | The Prestige | 0.286 | Dir: Christopher Nolan, Genre: Drama, Genre: Thriller |
| 4 | Interstellar | 0.232 | Dir: Christopher Nolan, Genre: Drama, Cast: Michael Caine |
| 5 | Insomnia | 0.203 | Dir: Christopher Nolan, Genre: Crime, Genre: Thriller |

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/adreejaa-dash/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the missing dataset file
The `tmdb_5000_credits.csv` file (~38 MB) must be downloaded manually:

1. Go to [Kaggle: TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
2. Download both `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`
3. Place them in the `data/` directory:

```
data/
  tmdb_5000_movies.csv    ← already in repo
  tmdb_5000_credits.csv   ← download from Kaggle
```

### 4. (Optional) Set TMDB API key for real movie posters
```bash
cp .env.example .env
# Edit .env and add your TMDB API key
# Get a free key at: https://www.themoviedb.org/settings/api
```

### 5. Launch the app
```bash
streamlit run app.py
```

The app will be available at **http://localhost:8501**

### 6. (Optional) Run evaluation
```bash
python3 evaluation.py
```

---

## Module Overview

| File | Purpose |
|------|---------|
| `data_preprocessing.py` | Load, merge, parse, and clean TMDB CSVs (4,803 movies) |
| `feature_engineering.py` | Build weighted tag strings; TF-IDF / CountVec vectorization |
| `recommender.py` | `ContentRecommender` + `CollaborativeRecommender` + explainability |
| `ratings_store.py` | SQLite ratings persistence + user preference profile builder |
| `evaluation.py` | Genre-overlap Precision@k; TF-IDF vs CountVec comparison |
| `app.py` | Streamlit UI — search, recommendations, ratings, personalization |
| `generate_sample_data.py` | Generates 30-movie test CSVs (dev/CI use only) |

---

## Key Features

- ✅ **4,803 movies** from the real TMDB 5000 Kaggle dataset
- ✅ **Explainable recommendations** — shows shared genres, cast, director, keywords per result
- ✅ **Dual vectorization** — switch between TF-IDF and CountVectorizer in the sidebar
- ✅ **Collaborative filtering** — item-based CF via cosine similarity on ratings matrix
- ✅ **User ratings** — 1–5 star ratings stored in SQLite; builds a taste profile
- ✅ **Personalization** — re-ranks content-based results using your rating history
- ✅ **Movie posters** — fetched from TMDB API (graceful fallback placeholder)
- ✅ **"My Recommendations"** — aggregates recs from your highest-rated movies
- ✅ **No secrets hardcoded** — API keys via `.env` (gitignored)

---

## Evaluation Results

Genre-overlap Precision@10 (higher = more genre-consistent):

| Movie | TF-IDF P@10 | CountVec P@10 |
|-------|-------------|---------------|
| The Dark Knight | 1.00 | 1.00 |
| Inception | 1.00 | 1.00 |
| The Godfather | 1.00 | 1.00 |
| Interstellar | 1.00 | 1.00 |

Run `python3 evaluation.py` to reproduce.

---

## Future Improvements

| Idea | Complexity | Impact |
|------|------------|--------|
| Matrix Factorization (SVD/ALS) | Medium | High — better CF quality |
| Deep learning embeddings (BERT/Sentence-T5) | High | High — semantic similarity |
| Hybrid model (weighted content + CF scores) | Medium | High |
| Real MovieLens dataset integration | Low | Medium |
| User authentication & multi-user profiles | Medium | Medium |
| TMDB API for full metadata enrichment | Low | Medium |
| Popularity decay / freshness weighting | Low | Low |
| A/B test TF-IDF vs CountVec recommendations | Low | Low |

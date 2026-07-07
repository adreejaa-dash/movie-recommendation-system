"""
app.py  —  Movie Recommendation System  —  Streamlit UI
Run: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# Load .env file if present (safe no-op if python-dotenv not installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from data_preprocessing import load_and_clean
from recommender import build_content_recommender, simulate_collab_recommender
from ratings_store import save_rating, get_ratings, build_user_profile

st.set_page_config(
    page_title="CineMatch — AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark cinematic theme
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0d0d1a;
    color: #e8e8f0;
}
.stApp { background: linear-gradient(135deg, #0d0d1a 0%, #12102a 60%, #0d1a2a 100%); }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(180deg, rgba(99,102,241,.18) 0%, transparent 100%);
    border-radius: 20px;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.hero p { color: #a0a0c0; font-size: 1.1rem; margin-top: .4rem; }

/* ── Movie Card ── */
.movie-card {
    background: linear-gradient(145deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 16px;
    padding: 1.1rem;
    margin-bottom: 1rem;
    transition: all .25s ease;
    backdrop-filter: blur(10px);
}
.movie-card:hover {
    border-color: rgba(99,102,241,.6);
    box-shadow: 0 0 24px rgba(99,102,241,.2);
    transform: translateY(-3px);
}
.movie-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8e8f8;
    margin-bottom: .3rem;
}
.sim-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    color: white;
    font-size: .75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.why-chip {
    display: inline-block;
    background: rgba(99,102,241,.18);
    border: 1px solid rgba(99,102,241,.35);
    color: #c4b5fd;
    font-size: .7rem;
    padding: 2px 8px;
    border-radius: 12px;
    margin: 2px;
}
.rating-stars { font-size: 1.3rem; }
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #a78bfa;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(167,139,250,.3);
    padding-bottom: .4rem;
}
.cf-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ec4899, #f97316);
    color: white;
    font-size: .75rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 20px;
    margin-bottom: .5rem;
}
.genre-tag {
    display: inline-block;
    background: rgba(236,72,153,.15);
    border: 1px solid rgba(236,72,153,.3);
    color: #f9a8d4;
    font-size: .7rem;
    padding: 2px 8px;
    border-radius: 10px;
    margin: 2px;
}
.no-data {
    background: rgba(239,68,68,.12);
    border: 1px solid rgba(239,68,68,.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: #fca5a5;
}
.stSelectbox > div { border-radius: 12px !important; }
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #a78bfa) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: .5rem 1.5rem !important;
    transition: opacity .2s ease !important;
}
.stButton > button:hover { opacity: .85 !important; }
.stSlider { padding-top: .3rem; }
div[data-testid="metric-container"] {
    background: rgba(99,102,241,.1);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 12px;
    padding: .8rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    return load_and_clean()


@st.cache_resource(show_spinner=False)
def get_content_rec(method="tfidf"):
    df = load_data()
    return build_content_recommender(df, method=method)


@st.cache_resource(show_spinner=False)
def get_collab_rec():
    df = load_data()
    return simulate_collab_recommender(df)


# ──────────────────────────────────────────────────────────────────────────────
# Poster fetching
# ──────────────────────────────────────────────────────────────────────────────

# poster_html removed — tmdb_5000_movies.csv has no poster_path column


# ──────────────────────────────────────────────────────────────────────────────
# Helper renderers
# ──────────────────────────────────────────────────────────────────────────────

def render_why(why: dict):
    chips = []
    if why.get("shared_director"):
        chips.append(f"🎬 {why['shared_director']}")
    for g in (why.get("shared_genres") or []):
        chips.append(f"🏷 {g}")
    for a in (why.get("shared_cast") or [])[:3]:
        chips.append(f"🎭 {a}")
    for k in (why.get("shared_keywords") or [])[:2]:
        chips.append(f"🔑 {k}")
    if chips:
        return " ".join(f'<span class="why-chip">{c}</span>' for c in chips)
    return '<span class="why-chip">✨ Similar vibe</span>'


def stars(rating: float) -> str:
    full = int(round(rating / 2))
    return "⭐" * full + "☆" * (5 - full)


def render_movie_card(row: pd.Series, card_id: str, show_rating_widget: bool = True):
    genres_html = " ".join(
        f'<span class="genre-tag">{g}</span>'
        for g in (row.get("genres") or [])[:4]
    )
    sim = row.get("similarity", row.get("cf_similarity", 0))
    sim_label = f"{sim:.0%}" if sim > 0 else ""
    badge_class = "sim-badge" if "similarity" in row else "cf-badge"
    why_html = render_why(row["why"]) if "why" in row and row["why"] else ""

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-title">{row['title']}</div>
        {"<span class='" + badge_class + "'>" + sim_label + " match</span>" if sim_label else ""}
        <div style="margin:.3rem 0;">{genres_html}</div>
        <div style="color:#a0a0b8; font-size:.8rem;">
            ⭐ {row.get('vote_average', 0):.1f} &nbsp;|&nbsp;
            🎬 {row.get('director','—')}
        </div>
        <div style="margin-top:.4rem;">{why_html}</div>
    </div>
    """, unsafe_allow_html=True)

    if show_rating_widget:
        r_col1, r_col2 = st.columns([2, 1])
        with r_col1:
            user_rating = st.slider(
                "Your rating",
                1.0, 5.0, 3.0, 0.5,
                key=f"rating_{card_id}",
                label_visibility="collapsed",
            )
        with r_col2:
            if st.button("Save ⭐", key=f"save_{card_id}"):
                mid = int(row.get("movie_id", 0))
                save_rating(mid, row["title"], user_rating)
                st.success("Saved!", icon="✅")
                st.cache_data.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    method = st.selectbox(
        "Vectorization Method",
        ["tfidf", "count"],
        format_func=lambda x: "TF-IDF (recommended)" if x == "tfidf" else "CountVectorizer",
        key="vec_method",
    )

    top_n = st.slider("Number of Recommendations", 5, 20, 10, key="top_n")

    show_collab = st.checkbox("Show Collaborative Filtering", value=True)

    st.markdown("---")
    st.markdown("### 📊 My Ratings")
    my_ratings = get_ratings()
    if my_ratings.empty:
        st.info("No ratings yet. Rate some movies!")
    else:
        st.dataframe(
            my_ratings[["title", "rating"]].rename(columns={"title": "Movie", "rating": "⭐"}),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='color:#6b6b8a; font-size:.75rem;'>CineMatch • Portfolio Project<br>"
        "TMDB 5000 Dataset</div>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# Main content
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🎬 CineMatch</h1>
    <p>AI-powered movie recommendations — discover films tailored to your taste</p>
</div>
""", unsafe_allow_html=True)

# Check for data files
movies_path = os.path.join("data", "tmdb_5000_movies.csv")
credits_path = os.path.join("data", "tmdb_5000_credits.csv")

if not (os.path.exists(movies_path) and os.path.exists(credits_path)):
    st.markdown("""
    <div class="no-data">
        <h3>📁 Dataset Not Found</h3>
        <p>Please download the <strong>TMDB 5000 Movie Dataset</strong> from Kaggle and place the files in the <code>data/</code> folder:</p>
        <ul style="text-align:left; display:inline-block;">
            <li><code>data/tmdb_5000_movies.csv</code></li>
            <li><code>data/tmdb_5000_credits.csv</code></li>
        </ul>
        <p><a href="https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata" target="_blank" style="color:#f9a8d4;">
            🔗 Download from Kaggle →
        </a></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load data + build recommenders
with st.spinner("🔄 Loading models..."):
    df = load_data()
    rec = get_content_rec(method)

# ── Movie Search ──
col_search, col_btn = st.columns([5, 1])
with col_search:
    search_query = st.text_input(
        "Search for a movie",
        placeholder="e.g. The Dark Knight, Inception, Avatar...",
        label_visibility="collapsed",
        key="search_input",
    )

# Filter titles based on search
all_titles = sorted(df["title"].tolist())
if search_query:
    filtered = [t for t in all_titles if search_query.lower() in t.lower()]
    if not filtered:
        filtered = all_titles
else:
    filtered = all_titles

selected_movie = st.selectbox(
    "Select Movie",
    filtered,
    label_visibility="collapsed",
    key="movie_select",
)

# ── Query Movie Info ──
if selected_movie:
    movie_row = df[df["title"] == selected_movie].iloc[0]

    st.markdown('<div class="section-header">📽️ Selected Movie</div>', unsafe_allow_html=True)

    genre_html = " ".join(
        f'<span class="genre-tag">{g}</span>'
        for g in (movie_row.get("genres") or [])
    )
    cast_html = ", ".join((movie_row.get("cast") or [])[:5])
    st.markdown(f"""
    <div class="movie-card">
        <div style="font-size:1.5rem; font-weight:900; color:#e8e8f8; margin-bottom:.4rem;">{selected_movie}</div>
        <div style="margin-bottom:.4rem;">{genre_html}</div>
        <div style="color:#a0a0b8; margin-bottom:.5rem;">
            ⭐ {movie_row.get('vote_average',0):.1f} &nbsp;|&nbsp;
            🎬 {movie_row.get('director','—')} &nbsp;|&nbsp;
            👥 {cast_html}
        </div>
        <div style="color:#c0c0d8; font-size:.9rem; line-height:1.6;">
            {(movie_row.get('overview','') or '')[:300]}{"..." if len(movie_row.get('overview','') or '') > 300 else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Content-Based Recommendations ──
    st.markdown(
        f'<div class="section-header">🤖 Content-Based Recommendations '
        f'<span style="font-size:.9rem; font-weight:400; color:#6b6b8a;">'
        f'({method.upper()})</span></div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Finding similar movies..."):
        # Load user profile for personalization
        user_profile = build_user_profile(movies_df=df)
        recs = rec.recommend(selected_movie, top_n=top_n, user_profile=user_profile or None)

    if recs.empty:
        st.warning("No recommendations found. Try a different movie.")
    else:
        # Merge genre/director info from df
        recs_merged = recs.merge(
            df[["movie_id", "vote_average"]].rename(columns={"vote_average": "_va"}),
            on="movie_id", how="left"
        )

        n_cols = 2
        rows = [recs.iloc[i:i+n_cols] for i in range(0, len(recs), n_cols)]
        for row_group in rows:
            cols = st.columns(n_cols)
            for col, (_, movie) in zip(cols, row_group.iterrows()):
                with col:
                    render_movie_card(movie, card_id=f"cb_{movie['title']}")

    # ── Collaborative Filtering ──
    if show_collab:
        st.markdown(
            '<div class="section-header">👥 Users Who Liked This Also Liked</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Running collaborative filtering..."):
            cf_rec = get_collab_rec()
            query_id = int(movie_row.get("movie_id", 0))
            cf_recs = cf_rec.recommend(query_id, top_n=6)

        if cf_recs.empty:
            st.info("Not enough data for collaborative recommendations for this movie.")
        else:
            cf_recs = cf_recs.merge(
                df[["movie_id", "genres", "cast", "director", "vote_average"]],
                on="movie_id", how="left"
            )
            cf_recs["why"] = cf_recs.apply(
                lambda r: {
                    "shared_genres": list(
                        set(movie_row.get("genres", []) or [])
                        & set(r.get("genres", []) or [])
                    ),
                    "shared_director": (
                        movie_row.get("director")
                        if movie_row.get("director") == r.get("director") else None
                    ),
                    "shared_cast": list(
                        set(movie_row.get("cast", []) or [])
                        & set(r.get("cast", []) or [])
                    ),
                    "shared_keywords": [],
                },
                axis=1,
            )
            cf_rows = [cf_recs.iloc[i:i+3] for i in range(0, len(cf_recs), 3)]
            for row_group in cf_rows:
                cols = st.columns(3)
                for col, (_, movie) in zip(cols, row_group.iterrows()):
                    with col:
                        render_movie_card(movie, card_id=f"cf_{movie['title']}")

    # ── My Personalized Recommendations ──
    my_ratings = get_ratings()
    if not my_ratings.empty:
        st.markdown(
            '<div class="section-header">❤️ My Personalized Picks</div>',
            unsafe_allow_html=True,
        )
        st.caption("Based on your ratings — re-ranked by your taste profile.")

        # Aggregate recs from top-rated movies
        liked = my_ratings[my_ratings["rating"] >= 4.0].head(3)
        personal_recs = []
        seen = set()
        for _, r in liked.iterrows():
            movie_title_rated = r["title"]
            if movie_title_rated not in df["title"].values:
                continue
            p_recs = rec.recommend(
                movie_title_rated, top_n=5, user_profile=user_profile
            )
            for _, pr in p_recs.iterrows():
                if pr["title"] not in seen and pr["title"] != selected_movie:
                    personal_recs.append(pr)
                    seen.add(pr["title"])

        if personal_recs:
            personal_df = pd.DataFrame(personal_recs).sort_values(
                "similarity", ascending=False
            ).head(6)
            p_rows = [personal_df.iloc[i:i+3] for i in range(0, len(personal_df), 3)]
            for row_group in p_rows:
                cols = st.columns(3)
                for col, (_, movie) in zip(cols, row_group.iterrows()):
                    with col:
                        render_movie_card(movie, card_id=f"my_{movie['title']}")
        else:
            st.info("Rate more movies to see personalized picks here!")

from data_preprocessing import load_and_clean
from recommender import build_content_recommender

df = load_and_clean()
print(f"Loaded {len(df)} movies from real Kaggle data")
print(f"Columns: {list(df.columns)}")
print()

rec = build_content_recommender(df, method="tfidf")
results = rec.recommend("The Dark Knight", top_n=10)
print('Top 10 recs for "The Dark Knight":')
print(f"{'#':<4} {'Title':<45} {'Sim':>6}  Shared Features")
print("-" * 90)
for i, (_, r) in enumerate(results.iterrows(), 1):
    why = r["why"]
    shared = []
    if why.get("shared_director"):
        shared.append(f"Dir:{why['shared_director']}")
    for g in why.get("shared_genres", [])[:2]:
        shared.append(f"Genre:{g}")
    for a in why.get("shared_cast", [])[:1]:
        shared.append(f"Cast:{a}")
    print(f"{i:<4} {r['title']:<45} {r['similarity']:>6.3f}  {shared[:3]}")

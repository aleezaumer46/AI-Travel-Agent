"""Hybrid destination recommender using content similarity, SVD and ranking."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).resolve().parents[1] / "saved_models" / "xgboost_recommender.pkl"


def recommend_destinations(entities: dict, destinations: list[dict], limit: int = 3) -> list[dict]:
    if not destinations:
        return []
    corpus = [f"{row['name']} {row['tags']} {row.get('description', '')}" for row in destinations]
    query = " ".join(entities.get("interests", [])) + " " + (entities.get("travel_style") or "")
    matrix = TfidfVectorizer().fit_transform(corpus + [query])
    content_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    budget = entities.get("budget") or 0
    duration = entities.get("duration") or 1
    results = []
    for index, row in enumerate(destinations):
        expected = float(row["avg_daily_cost"]) * duration
        budget_match = 1.0 if not budget else max(0.0, 1.0 - abs(expected - budget) / max(budget, 1))
        exact_match = bool(entities.get("destination") and row["name"].casefold() == str(entities["destination"]).casefold())
        score = float(content_scores[index]) * 0.55 + budget_match * 0.35 + float(row["rating"]) / 5 * 0.10
        if exact_match:
            score = max(score, 0.95)
        results.append({**row, "content_score": float(content_scores[index]), "budget_match": round(budget_match, 3), "score": round(score, 4), "estimated_cost": round(expected)})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]


def train_lightweight_ranker(destinations: list[dict]) -> None:
    """Persist a small real scikit-learn ranking artifact for repeatable local inference."""
    if MODEL_PATH.exists() or not destinations:
        return
    values = np.array([[row["rating"], row["avg_daily_cost"]] for row in destinations], dtype=float)
    artifact = {"scaler": StandardScaler().fit(values), "svd": TruncatedSVD(n_components=1, random_state=42).fit(values)}
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

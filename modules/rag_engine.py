"""Local grounded retrieval over the bundled travel dataset."""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "travel_data.csv"


def load_documents() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    return pd.read_csv(DATA_PATH).fillna("").to_dict("records")


def retrieve(query: str, k: int = 4) -> list[dict]:
    documents = load_documents()
    if not documents or not query.strip():
        return []
    texts = [f"{row['name']} {row['province_country']} {row['tags']} {row['description']}" for row in documents]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = np.argsort(scores)[::-1] if scores.size else []
    matches = [{**documents[index], "retrieval_score": float(scores[index])} for index in ranked[:k] if scores[index] > 0]
    if not matches:
        return [{**documents[index], "retrieval_score": 0.0} for index in ranked[:k]]
    return matches


def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant local travel knowledge was found. Do not make unsupported factual claims."
    return "\n".join(f"- {chunk['name']}: {chunk['description']} Tags: {chunk['tags']}. Typical daily cost: PKR {chunk['avg_daily_cost']}." for chunk in chunks)

"""Local hotel search and ranking for the Travel Hub."""
import csv
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hotels.csv"
DESTINATION_COORDINATES = {
    "Hunza": (36.3167, 74.6500), "Skardu": (35.2970, 75.6330), "Swat": (34.7717, 72.3600),
    "Lahore": (31.5204, 74.3587), "Islamabad": (33.6844, 73.0479), "Naran Kaghan": (34.9080, 73.6500),
    "Chitral": (35.8500, 71.7860),
}


def load_hotels() -> list[dict]:
    with DATA_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _distance_km(latitude: float, longitude: float, origin: tuple[float, float]) -> float:
    lat1, lon1 = radians(origin[0]), radians(origin[1])
    lat2, lon2 = radians(latitude), radians(longitude)
    value = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(value))


def recommend_hotels(destination: str, hotels: list[dict], budget: int | None = None, limit: int = 5) -> list[dict]:
    origin = DESTINATION_COORDINATES.get(destination)
    if not origin:
        return []
    results = []
    for hotel in hotels:
        if hotel["destination"].casefold() != destination.casefold():
            continue
        item = dict(hotel)
        item["distance_km"] = round(_distance_km(float(item["latitude"]), float(item["longitude"]), origin), 1)
        item["budget_fit"] = 1.0 if not budget else max(0.0, 1.0 - abs(float(item["price_per_night"]) - budget) / max(budget, 1))
        distance_score = max(0.0, 1.0 - item["distance_km"] / 15)
        rating_score = float(item["guest_rating"]) / 5
        value_score = min(1.0, (float(item["guest_rating"]) / 5) * (12000 / max(float(item["price_per_night"]), 1)))
        item["recommendation_score"] = round(distance_score * 0.30 + rating_score * 0.40 + value_score * 0.20 + item["budget_fit"] * 0.10, 4)
        results.append(item)
    return sorted(results, key=lambda item: item["recommendation_score"], reverse=True)[:limit]

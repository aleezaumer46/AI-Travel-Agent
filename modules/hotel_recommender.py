"""Local hotel search and ranking for the Travel Hub."""
import csv
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hotels.csv"
DESTINATION_COORDINATES = {
    "Hunza": (36.3167, 74.6500), "Skardu": (35.2970, 75.6330), "Gilgit": (35.9208, 74.3083),
    "Nagar Valley": (36.2500, 74.4500), "Astore": (35.3667, 74.8667), "Deosai Plains": (35.0000, 75.2500),
    "Fairy Meadows": (35.4000, 74.6000), "Khunjerab Pass": (36.8500, 75.4167), "Shigar": (35.4200, 75.7300),
    "Swat": (34.7717, 72.3600), "Kalam": (35.4900, 72.5800), "Malam Jabba": (35.2120, 72.5630),
    "Naran Kaghan": (34.9080, 73.6500), "Murree": (33.9070, 73.3900), "Shogran": (34.6300, 73.5000),
    "Babusar Pass": (34.7600, 73.5700), "Chitral": (35.8500, 71.7860), "Kalash Valley": (35.7000, 71.7500),
    "Islamabad": (33.6844, 73.0479), "Rawalpindi": (33.5651, 73.0169), "Taxila": (33.7460, 72.8397),
    "Lahore": (31.5204, 74.3587), "Multan": (30.1575, 71.5249), "Bahawalpur": (29.3956, 71.6836),
    "Derawar Fort": (28.7750, 71.3350), "Peshawar": (34.0151, 71.5249), "Abbottabad": (34.1688, 73.2215),
    "Mingora": (34.7717, 72.3600), "Quetta": (30.1798, 66.9750), "Ziarat": (30.3820, 67.7250),
    "Karachi": (24.8607, 67.0011), "Gwadar": (25.1264, 62.3225), "Hingol National Park": (25.5000, 65.5000),
    "Ormara": (25.2100, 64.6350), "Thatta": (24.7475, 67.9235), "Moenjodaro": (27.3290, 68.1386),
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

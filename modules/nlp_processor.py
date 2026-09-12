"""Fast multilingual intent and entity extraction for travel requests."""
import re
from typing import Any

DESTINATIONS = ["Hunza", "Skardu", "Swat", "Lahore", "Islamabad", "Naran Kaghan", "Chitral"]
INTERESTS = ["mountains", "nature", "photography", "adventure", "lakes", "family", "history", "food", "culture", "city", "waterfalls"]
STYLE_WORDS = {"budget": "budget", "luxury": "luxury", "comfortable": "comfort", "comfort": "comfort", "adventure": "adventure", "family": "family"}
ROMAN_MAP = {"pahaar": "mountains", "jheel": "lakes", "fitrat": "nature", "khaana": "food", "khana": "food", "tareekh": "history", "khandan": "family", "safar": "travel", "din": "days", "rupay": "PKR"}


def standardize_text(text: str) -> str:
    value = text.lower().strip()
    for source, target in ROMAN_MAP.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    value = value.replace("lakh", "100000").replace("lac", "100000")
    value = re.sub(r"(\d+)k\b", lambda match: str(int(match.group(1)) * 1000), value)
    return re.sub(r"\s+", " ", value)


def _number_after(patterns: list[str], text: str) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def extract_entities(text: str, language: str = "English") -> dict[str, Any]:
    normalized = standardize_text(text)
    destination = next((item for item in DESTINATIONS if item.lower() in normalized), None)
    budget = _number_after([r"(?:budget|under|around|with)\s*(?:of\s*)?(?:pkr\s*)?([\d,]+)", r"(?:pkr|rs)\s*([\d,]+)", r"([\d,]+)\s*(?:budget|pkr|rs)"], normalized)
    duration = _number_after([r"(\d+)\s*(?:days?|din)", r"for\s*(\d+)"], normalized)
    interests = [item for item in INTERESTS if re.search(rf"\b{re.escape(item)}\b", normalized)]
    style = next((value for key, value in STYLE_WORDS.items() if re.search(rf"\b{key}\b", normalized)), None)
    missing = []
    if not destination:
        missing.append("destination")
    if not duration:
        missing.append("duration")
    if not budget:
        missing.append("budget")
    return {"destination": destination, "budget": budget, "duration": duration, "interests": interests, "travel_style": style, "language": language, "missing": missing, "normalized_text": normalized}


def classify_intent(text: str) -> str:
    normalized = standardize_text(text)
    if any(word in normalized for word in ("recommend", "suggest", "plan", "trip", "travel")):
        return "plan_trip"
    if any(word in normalized for word in ("weather", "forecast", "temperature")):
        return "check_weather"
    return "general_travel_question"

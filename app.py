"""AI Travel Agent - tabbed Streamlit experience."""
import hashlib
import io
import textwrap
import csv
from math import asin, cos, radians, sin, sqrt
import requests
import streamlit as st
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from modules import database

destination_rows = database.destination_rows
favorite_rows = database.favorite_rows
initialize_database = database.initialize_database
log_trip = database.log_trip
save_favorite = database.save_favorite
add_expense = database.add_expense
expense_rows = database.expense_rows
save_place = database.save_place
saved_place_rows = database.saved_place_rows
save_checklist_item = database.save_checklist_item
checklist_rows = database.checklist_rows
set_checklist_item = database.set_checklist_item
add_document = database.add_document
document_rows = database.document_rows
add_reminder = database.add_reminder
reminder_rows = database.reminder_rows
def _booking_feature_unavailable(_booking: dict) -> str:
    raise RuntimeError("The deployed database module is outdated. Redeploy the latest GitHub commit.")


save_transport_booking = getattr(database, "save_transport_booking", _booking_feature_unavailable)
transport_booking_rows = getattr(database, "transport_booking_rows", lambda: [])
from modules.hotel_recommender import DESTINATION_COORDINATES, recommend_hotels
from modules.nlp_processor import classify_intent, extract_entities
from modules.rag_engine import format_context, load_documents, retrieve
from modules.recommender import recommend_destinations, train_lightweight_ranker
from modules.vision import describe_image, recognize_landmark

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")


def get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    secrets_path = Path(__file__).resolve().parent / ".streamlit" / "secrets.toml"
    try:
        if secrets_path.exists():
            with secrets_path.open("rb") as file:
                data = tomllib.load(file)
            value = data.get(name)
            if value:
                return value
    except Exception:
        pass
    return None


def load_hotel_directory() -> list[dict]:
    data_path = Path(__file__).resolve().parent / "data" / "hotels.csv"
    with data_path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def get_weather(destination: str) -> str:
    key = get_secret("WEATHER_API_KEY")
    if not key:
        return "Weather API key is not configured."
    try:
        response = requests.get("https://api.openweathermap.org/data/2.5/weather", params={"q": destination, "appid": key, "units": "metric"}, timeout=5)
        response.raise_for_status()
        data = response.json()
        return f"{data['weather'][0]['description'].title()} | {data['main']['temp']:.0f}°C | Feels like {data['main']['feels_like']:.0f}°C"
    except requests.RequestException:
        return "Weather service is temporarily unavailable."


def generate_itinerary(entities: dict, context: str, weather: str) -> str:
    key = get_secret("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            response = OpenAI(api_key=key).chat.completions.create(model=get_secret("OPENAI_MODEL") or "gpt-4o-mini", temperature=0.2, messages=[{"role": "system", "content": "Create a concise day-by-day travel itinerary. Use only supplied context for factual claims and mention missing information."}, {"role": "user", "content": f"Trip: {entities}\nLocal context:\n{context}\nWeather: {weather}\nReturn markdown with a realistic budget allocation."}])
            return response.choices[0].message.content
        except Exception:
            pass
    daily = max(1, int(entities["budget"] / max(entities["duration"], 1)))
    days = "\n".join(f"### Day {day}\n- Morning: Start a relaxed exploration of {entities['destination']}.\n- Afternoon: Choose a {', '.join(entities.get('interests') or ['local culture'])} activity.\n- Evening: Keep spending near PKR {daily:,} and retain a travel buffer." for day in range(1, entities["duration"] + 1))
    return f"## {entities['duration']}-Day {entities['destination']} Plan\n\n**Budget:** PKR {entities['budget']:,}  \n**Weather:** {weather}\n\n### Grounded local context\n{context}\n\n{days}"


def budget_breakdown(budget: int, duration: int) -> dict[str, int]:
    total = max(0, int(budget))
    allocation = {"Stay": .32, "Transport": .25, "Food": .18, "Activities": .15, "Emergency buffer": .10}
    result = {name: int(total * share) for name, share in allocation.items()}
    result["Emergency buffer"] += total - sum(result.values())
    return result


def hotel_budget_per_night(budget: int, duration: int) -> int:
    """Use the stay allocation when matching hotels to the total trip budget."""
    return budget_breakdown(budget, duration)["Stay"] // max(duration, 1)


def packing_list(entities: dict) -> list[str]:
    items = ["CNIC/passport, copies and emergency contacts", "Phone, charger and power bank", "Reusable water bottle", "Basic medicines and first-aid kit", "Comfortable walking shoes", "Weather-appropriate layers"]
    interests = set(entities.get("interests", []))
    if interests & {"mountains", "adventure", "lakes"}:
        items += ["Warm waterproof jacket", "Sun protection and sunglasses", "Small daypack"]
    if "photography" in interests:
        items += ["Camera/phone storage and spare battery"]
    if "family" in interests:
        items += ["Small snacks and child-friendly essentials"]
    return items


def food_recommendations(destination: str, interests: list[str]) -> list[str]:
    food_map = {
        "Lahore": ["Lahori chargha", "Nihari", "Hareesa", "Food Street tasting"],
        "Islamabad": ["Saidpur Village cuisine", "Pakistani barbecue", "Margalla tea stops"],
        "Hunza": ["Chapshuro", "Hunza bread", "Apricot juice", "Local walnut cake"],
        "Skardu": ["Balti cuisine", "Apricot soup", "Momos", "Local trout"],
    }
    return food_map.get(destination, [f"Try verified local specialties in {destination}", "Ask your hotel for hygienic family restaurants", "Keep one flexible meal for local discovery"])


def convert_currency(amount: float, source: str, target: str) -> float:
    rates = {"PKR": 1.0, "USD": 0.0036, "EUR": 0.0033, "GBP": 0.0028, "AED": 0.0132, "TRY": 0.12}
    return amount * rates.get(source, 1.0) / rates.get(target, 1.0)


def local_phrases(language: str) -> list[tuple[str, str]]:
    phrase_sets = {"Urdu": [("Hello", "Assalam-o-alaikum"), ("How much?", "Yeh kitne ka hai?"), ("Help", "Madad kijiye"), ("Where is the hotel?", "Hotel kahan hai?")], "Arabic": [("Hello", "Marhaba"), ("How much?", "Kam al-sear?"), ("Help", "Musaada"), ("Where is the hotel?", "Ayna al-funduq?")], "Turkish": [("Hello", "Merhaba"), ("How much?", "Ne kadar?"), ("Help", "Yardim edin"), ("Where is the hotel?", "Otel nerede?")]}
    return phrase_sets.get(language, [("Hello", "Hello"), ("How much?", "How much?"), ("Help", "Please help"), ("Where is the hotel?", "Where is the hotel?")])


def is_roman_urdu(text: str) -> bool:
    roman_words = {"ma", "mein", "ka", "ki", "ke", "kahan", "kon", "sa", "hai", "ha", "mujhy", "mujhe", "chahiye", "kitna", "kitne", "acha", "qareeb", "sab", "se", "batao", "karo", "karna"}
    words = set(re.findall(r"[a-z]+", text.casefold()))
    return len(words & roman_words) >= 2


def translate_phrase(text: str, target_language: str) -> tuple[str, str]:
    """Translate a travel phrase with OpenAI and keep useful offline fallbacks."""
    phrase = text.strip()
    if not phrase:
        return "", "Enter a phrase to translate."
    roman_input = is_roman_urdu(phrase)
    requested_language = target_language
    if roman_input:
        target_language = "English"
    if target_language == "English" and not roman_input:
        return phrase, "English is already the selected language."
    key = get_secret("OPENAI_API_KEY")
    if key:
        try:
            from openai import OpenAI
            response = OpenAI(api_key=key).chat.completions.create(
                model=get_secret("OPENAI_MODEL") or "gpt-4o-mini",
                temperature=0,
                max_tokens=180,
                messages=[
                    {"role": "system", "content": f"Translate the user's travel phrase into {target_language}. Return only the translation, with no explanation."},
                    {"role": "user", "content": phrase},
                ],
            )
            translated = (response.choices[0].message.content or "").strip()
            if translated:
                return translated, "Translated with the configured AI translation service."
        except Exception:
            pass
    fallback_phrases = {
        "English": {
            "islamabad ma kon sa best hotel ha": "Which is the best hotel in Islamabad?",
            "islamabad mein kon sa best hotel hai": "Which is the best hotel in Islamabad?",
            "how much does this taxi cost": "How much does this taxi cost?",
            "mujhe aik kamra chahiye": "I need a room.",
            "hotel kahan hai": "Where is the hotel?",
        },
        "Urdu": {
            "how much does this taxi cost?": "Yeh taxi kitne ki hai?",
            "how much?": "Yeh kitne ka hai?",
            "where is the hotel?": "Hotel kahan hai?",
            "please help me": "Barah-e-karam meri madad karein.",
            "i need a room": "Mujhe aik kamra chahiye.",
            "where is the bathroom?": "Bathroom kahan hai?",
        },
        "Arabic": {
            "how much does this taxi cost?": "كم تكلفة سيارة الأجرة؟",
            "how much?": "كم السعر؟",
            "where is the hotel?": "أين الفندق؟",
            "please help me": "من فضلك ساعدني.",
        },
        "Turkish": {
            "how much does this taxi cost?": "Bu taksi ne kadar?",
            "how much?": "Ne kadar?",
            "where is the hotel?": "Otel nerede?",
            "please help me": "Lütfen bana yardım edin.",
        },
    }
    translated = fallback_phrases.get(target_language, {}).get(phrase.casefold())
    if translated:
        return translated, "Translated with the offline travel phrasebook."
    if roman_input and requested_language != "English":
        return phrase, "Roman Urdu detected. Add an OpenAI API key for free-form Roman Urdu to English translation."
    return phrase, f"No offline match found. Add an OpenAI API key for free-form {target_language} translation."


def safety_brief(destination: str) -> list[str]:
    return [f"Save local emergency numbers for {destination} before departure.", "Use registered transport and agree the fare before moving.", "Keep a digital copy of passport, visa and insurance separately.", "Avoid isolated routes after dark and share your live route with a trusted contact.", "Common scam signals: urgent payment requests, fake guides, over-friendly diversions and unmetered fares."]


def daily_trip_schedule(entities: dict) -> str:
    """Create a concrete daily schedule from the same local tools used in the planner."""
    destination = entities["destination"]
    duration = int(entities.get("duration") or 1)
    budget = int(entities.get("budget") or 0)
    hotels = recommend_hotels(destination, load_hotel_directory(), hotel_budget_per_night(budget, duration), 1)
    hotel = hotels[0] if hotels else None
    hotel_name = hotel["name"] if hotel else "Choose a verified hotel in the destination"
    stay_summary = (
        f"**Hotel:** {hotel_name}  \n"
        f"**Rent:** PKR {float(hotel['price_per_night']):,.0f}/night · "
        f"PKR {float(hotel['price_per_night']) * duration:,.0f} for {duration} nights  \n"
        f"**Contact:** {hotel['contact_number']} · **Address:** {hotel['address']}"
        if hotel else "**Hotel:** No matching local hotel found; confirm accommodation before booking."
    )
    places = nearby_place_details(destination)
    foods = food_recommendations(destination, entities.get("interests", []))
    daily_budget = budget // max(duration, 1)
    lines = [f"## {duration}-Day {destination} Schedule", f"**Total trip budget:** PKR {budget:,}  \n{stay_summary}"]
    for day in range(1, duration + 1):
        primary = places[(day - 1) % len(places)] if places else {"name": destination, "description": f"Explore {destination}", "visit": "Day visit"}
        secondary = places[day % len(places)] if len(places) > 1 else primary
        food = foods[(day - 1) % len(foods)]
        lines.append(
            f"### Day {day}: {destination} · {primary['name']} area\n"
            f"- **Stay:** {hotel_name}\n"
            f"- **Morning:** Visit **{primary['name']}**. {primary['description']}\n"
            f"- **Afternoon:** Explore **{secondary['name']}** and allow time for photos and local discovery.\n"
            f"- **Food:** Try {food}.\n"
            f"- **Evening:** Return to {hotel_name}, review the next day's route and keep an emergency buffer.\n"
            f"- **Estimated daily spend:** PKR {daily_budget:,}"
        )
    return "\n\n".join(lines)


def full_trip_plan(entities: dict, itinerary: str, weather: str) -> str:
    """Combine the planner, Trip Tools, and Travel Hub outputs into one plan."""
    destination = entities["destination"]
    budget = int(entities.get("budget") or 0)
    duration = int(entities.get("duration") or 1)
    breakdown = budget_breakdown(budget, duration)
    hotels = recommend_hotels(destination, load_hotel_directory(), hotel_budget_per_night(budget, duration), 3)
    origin = entities.get("origin") or "Current location"
    travelers = int(entities.get("travelers") or 1)
    transport = transport_options(origin, destination, travelers, entities.get("trip_type") or "One-way", budget)
    places = nearby_place_details(destination)[:4]
    foods = food_recommendations(destination, entities.get("interests", []))
    packing = packing_list(entities)
    guide = travel_guide(destination, entities)
    daily_schedule = daily_trip_schedule(entities)
    alternatives = [item for item in recommend_destinations(entities, destinations, limit=4) if item["name"].casefold() != destination.casefold()][:3]
    hotel_lines = "\n".join(
        f"- **{hotel['name']}** | PKR {float(hotel['price_per_night']):,.0f}/night | "
        f"PKR {float(hotel['price_per_night']) * duration:,.0f} for {duration} nights | "
        f"{float(hotel['guest_rating']):.1f}/5 ({int(hotel['review_count'])} reviews)\n"
        f"  Address: {hotel['address']} | Contact: {hotel['contact_number']}\n"
        f"  Amenities: {hotel['amenities'].replace(' | ', ', ')}"
        for hotel in hotels
    ) or "- No local hotel directory result; confirm accommodation separately."
    transport_lines = "\n".join(f"- {option['mode']}: PKR {option['cost']:,} estimated | {option['hours']:.1f} hours | {option['distance_km']:,} km" for option in transport)
    place_lines = "\n".join(f"- **{place['name']}**: {place['description']} ({place['visit']})" for place in places)
    alternative_lines = "\n".join(f"- {item['name']} | Fit {item['score']:.0%} | Estimated PKR {item['estimated_cost']:,}" for item in alternatives) or "- No alternative destination matches the current filters."
    return f"""{daily_schedule}

## Complete Trip Toolkit

### Destination travel guide
{guide}

### Alternative destinations
{alternative_lines}

### Budget allocation
""" + "\n".join(f"- {category}: PKR {amount:,}" for category, amount in breakdown.items()) + f"""

### Transport plan
**Route:** {origin} to {destination} · **Travellers:** {travelers} · **Trip type:** {entities.get('trip_type') or 'One-way'}
{transport_lines}

### Recommended stay
{hotel_lines}

### Nearby places to explore
{place_lines}

### Food plan
""" + "\n".join(f"- {food}" for food in foods) + "\n\n### Packing checklist\n" + "\n".join(f"- {item}" for item in packing) + f"\n\n**Weather:** {weather}\n\n*Confirm live prices, hotel availability, transport schedules and attraction timings before booking.*"


def compare_destinations(names: list[str], budget: int, duration: int) -> list[dict]:
    rows = {row["name"]: row for row in destinations}
    result = []
    for name in names:
        row = rows.get(name)
        if row:
            result.append({**row, "trip_estimate": int(float(row["avg_daily_cost"]) * duration), "budget_fit": max(0, min(100, round((1 - abs(float(row["avg_daily_cost"]) * duration - budget) / max(budget, 1)) * 100)))} )
    return result


def transport_options(origin: str, destination: str, travelers: int, trip_type: str, budget: int) -> list[dict]:
    coordinates = {**DESTINATION_COORDINATES, "Current location": (33.6844, 73.0479)}
    start = coordinates.get(origin, coordinates["Current location"])
    end = coordinates.get(destination, coordinates["Current location"])
    lat_gap = radians(end[0] - start[0])
    lon_gap = radians(end[1] - start[1])
    distance = round(6371 * 2 * asin(sqrt(sin(lat_gap / 2) ** 2 + cos(radians(start[0])) * cos(radians(end[0])) * sin(lon_gap / 2) ** 2)))
    multiplier = 2 if trip_type == "Return trip" else 1
    same_city = origin == destination
    options = []
    if same_city:
        options.append({"mode": "Cab / Taxi", "best_for": "Short city travel", "cost": 900 * multiplier, "hours": 0.5, "note": "Driver and fare are confirmed after the booking request."})
    else:
        options.extend([
            {"mode": "Cab / Taxi", "best_for": "Private door-to-door travel", "cost": max(3500, round(distance * 75 + 1500)) * multiplier, "hours": max(1.0, distance / 50), "note": "Driver, vehicle and final fare require provider confirmation."},
            {"mode": "Private car", "best_for": "Families and flexible stops", "cost": max(4500, round(distance * 65 + 2500)) * multiplier, "hours": max(1.0, distance / 55), "note": "Fuel, tolls and driver terms can change the quote."},
            {"mode": "Bus or coach", "best_for": "Lowest intercity cost", "cost": max(800, round(distance * 5.5)) * travelers * multiplier, "hours": max(2.0, distance / 45), "note": "Confirm operator schedule and seat availability."},
            {"mode": "Domestic flight", "best_for": "Long-distance routes", "cost": max(8000, round(distance * 12)) * travelers * multiplier, "hours": max(1.0, distance / 600 + 2), "note": "Fare excludes airport transfers and baggage changes."},
        ])
    for option in options:
        option["distance_km"] = distance
        option["per_person"] = round(option["cost"] / max(travelers, 1))
        option["budget_fit"] = max(0, min(100, round((1 - abs(option["cost"] - budget) / max(budget, 1)) * 100))) if budget else None
    return options


def nearby_places(destination: str) -> list[str]:
    nearby = {"Lahore": ["Badshahi Mosque", "Lahore Fort", "Walled City", "Food Street"], "Islamabad": ["Faisal Mosque", "Daman-e-Koh", "Pakistan Monument", "Saidpur Village"], "Hunza": ["Attabad Lake", "Baltit Fort", "Altit Fort", "Passu Cones"], "Skardu": ["Shangrila Resort", "Upper Kachura Lake", "Deosai Plains", "Mansehra viewpoints"]}
    return nearby.get(destination, [f"Verified local attractions near {destination}", "Local market or cultural center", "A nearby viewpoint", "A recommended day trip"])


def nearby_place_details(destination: str) -> list[dict[str, str]]:
    details = {
        "Lahore": {
            "Badshahi Mosque": ("https://images.unsplash.com/photo-1584551246679-0daf3d275d0f?auto=format&fit=crop&w=900&q=80", "Mughal-era mosque beside Lahore Fort.", "Old Lahore · 1-2 hours · Best near sunset", "Respect prayer times and dress modestly."),
            "Lahore Fort": ("https://images.unsplash.com/photo-1593693397690-362cb9666fc2?auto=format&fit=crop&w=900&q=80", "Historic citadel with palaces, courtyards and gateways.", "Walled City · 2-3 hours · Go early", "Check opening days and carry water."),
            "Walled City": ("https://images.unsplash.com/photo-1539650116574-75c0c6d73f6e?auto=format&fit=crop&w=900&q=80", "Atmospheric lanes filled with heritage, markets and local food.", "Central Lahore · 2-4 hours · Walking route", "Use a local guide and keep valuables secure."),
            "Food Street": ("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?auto=format&fit=crop&w=900&q=80", "A lively food stop for Lahori barbecue and traditional dishes.", "Fort Road · 1-2 hours · Evening", "Confirm restaurant prices before ordering."),
        },
        "Islamabad": {
            "Faisal Mosque": ("https://images.unsplash.com/photo-1584551246679-0daf3d275d0f?auto=format&fit=crop&w=900&q=80", "An iconic modern mosque framed by the Margalla Hills.", "E-8 · 1 hour · Morning or sunset", "Follow visitor areas and prayer-time guidance."),
            "Daman-e-Koh": ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=900&q=80", "A hilltop viewpoint overlooking Islamabad.", "Margalla Hills · 2-3 hours · Clear weather", "Roads can be busy on weekends; carry water."),
            "Pakistan Monument": ("https://images.unsplash.com/photo-1539650116574-75c0c6d73f6e?auto=format&fit=crop&w=900&q=80", "A national monument and museum on Shakarparian Hills.", "Shakarparian · 1-2 hours · Afternoon", "Verify museum hours before visiting."),
            "Saidpur Village": ("https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?auto=format&fit=crop&w=900&q=80", "A restored heritage village with cafés and hill views.", "Margalla foothills · 1-2 hours · Evening", "Expect crowded parking at peak times."),
        },
        "Skardu": {
            "Shangrila Resort": ("https://images.unsplash.com/photo-1464278533981-50106e6176b1?auto=format&fit=crop&w=900&q=80", "A lakeside resort near Lower Kachura Lake with mountain scenery and peaceful viewpoints.", "Lower Kachura · Half day · Daylight", "Confirm boating and resort access before travelling."),
            "Upper Kachura Lake": ("https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=900&q=80", "A clear alpine lake surrounded by rugged peaks and pine-covered slopes.", "Kachura valley · 2-4 hours · Morning", "Carry water, warm layers and cash for local transport."),
            "Deosai Plains": ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=900&q=80", "A high-altitude plateau known for wide landscapes, wildlife and dramatic mountain horizons.", "Day trip · Full day · Summer season", "Check road and weather conditions because access is seasonal."),
            "Mansehra viewpoints": ("https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?auto=format&fit=crop&w=900&q=80", "Scenic roadside viewpoints for a relaxed stop while travelling through the mountain route.", "Roadside stop · 1-2 hours · Clear weather", "Use marked viewpoints and avoid stopping on unsafe road edges."),
        },
        "Hunza": {
            "Attabad Lake": ("https://images.unsplash.com/photo-1464278533981-50106e6176b1?auto=format&fit=crop&w=900&q=80", "Turquoise alpine lake known for boating and dramatic views.", "Gojal · Half day · Daylight", "Wear layers and check road conditions."),
            "Baltit Fort": ("https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=900&q=80", "A centuries-old fort overlooking Karimabad and the valley.", "Karimabad · 1-2 hours · Morning", "The uphill walk needs comfortable shoes."),
            "Altit Fort": ("https://images.unsplash.com/photo-1516026672322-bc52d61a55d5?auto=format&fit=crop&w=900&q=80", "A heritage fort and village with wide valley viewpoints.", "Altit · 1-2 hours · Afternoon", "Allow time for the surrounding village walk."),
            "Passu Cones": ("https://images.unsplash.com/photo-1464278533981-50106e6176b1?auto=format&fit=crop&w=900&q=80", "Distinctive jagged peaks along the Karakoram Highway.", "Gojal · 1-2 hours · Clear daylight", "Keep distance from unstable roadside areas."),
        },
    }
    fallback_image = "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=900&q=80"
    places = nearby_places(destination)
    return [{"name": place, "image": details.get(destination, {}).get(place, (fallback_image, f"A recommended place to explore near {destination}.", "Nearby attraction · Check locally", "Confirm access, timings and current conditions before visiting."))[0], "description": details.get(destination, {}).get(place, ("", f"A recommended place to explore near {destination}.", "Nearby attraction · Check locally", "Confirm access, timings and current conditions before visiting."))[1], "visit": details.get(destination, {}).get(place, ("", "", "Nearby attraction · Check locally", "Confirm access, timings and current conditions before visiting."))[2], "tip": details.get(destination, {}).get(place, ("", "", "", "Confirm access, timings and current conditions before visiting."))[3]} for place in places]


def travel_guide(destination: str, entities: dict) -> str:
    context = format_context(retrieve(destination))
    return f"## Travel Guide: {destination}\n\n**Ideal trip length:** {entities.get('duration', duration_filter)} days  \n**Style:** {entities.get('travel_style', style_filter)}\n\n### Quick orientation\n{context}\n\n### Before you go\n- Confirm opening days, road conditions and current weather.\n- Keep original identification and copies secure.\n- Reserve a buffer for transport delays and local changes.\n\n### Nearby to explore\n" + "\n".join(f"- {place}" for place in nearby_places(destination))


def itinerary_pdf(markdown: str, title: str) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 18, 18)
    pdf.set_font("Helvetica", "B", 16)
    safe_title = title.encode("latin-1", "replace").decode("latin-1")
    pdf.multi_cell(pdf.epw, 10, safe_title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    for line in markdown.splitlines():
        safe_line = line.encode("latin-1", "replace").decode("latin-1")
        if safe_line.startswith("### "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            safe_line = safe_line[4:]
        elif safe_line.startswith("## "):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 14)
            safe_line = safe_line[3:]
        else:
            pdf.set_font("Helvetica", size=10)
            safe_line = safe_line.replace("**", "")
        chunks = []
        for paragraph in textwrap.wrap(safe_line, width=95, break_long_words=False, break_on_hyphens=False) or [""]:
            if len(paragraph) <= 120:
                chunks.append(paragraph)
            else:
                chunks.extend(paragraph[index:index + 80] for index in range(0, len(paragraph), 80))
        for chunk in chunks:
            pdf.multi_cell(pdf.epw, 6, chunk, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    return bytes(pdf.output())


def inject_theme() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    :root { --navy:#f4f3ef; --panel:#f7f6f3; --panel2:#edf1ee; --line:#dfe5e0; --text:#1f2323; --muted:#5f6968; --blue:#a7c2b8; --blue2:#a7c2b8; --green:#39c985; }
    .stApp { background:var(--navy); color:var(--text); }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#f9f9f7; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding:1.2rem 1.35rem; }
    [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] button svg { color:#19a9e8 !important; fill:#19a9e8 !important; stroke:#19a9e8 !important; opacity:1 !important; }
    div[data-testid="stSidebar"] .card { background:#e7ece9 !important; border:1px solid rgba(31,35,35,0.08) !important; }
    div[data-testid="stSidebar"] .card b, div[data-testid="stSidebar"] .card span { color:#1f2323 !important; }
    [data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stSidebarCollapseButton"] button:hover svg { color:#f5f7fb !important; fill:#f5f7fb !important; stroke:#f5f7fb !important; }
    h1,h2,h3,h4,p,label,span { color:var(--text); }
    h1 { font-size:2.35rem !important; letter-spacing:-.04em; }
    h2 { font-size:1.6rem !important; }
    .brand { display:flex; gap:.7rem; align-items:center; margin:.2rem 0 .3rem; }
    .brand-mark { font-size:2.1rem; }
    .brand-title { font-size:2rem; font-weight:700; letter-spacing:-.05em; }
    .subtitle { color:var(--muted); margin:.2rem 0 2.4rem; }
    .rule { height:1px; background:var(--line); margin:1.7rem 0; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:1.1rem; min-height:150px; }
    .card b { color:#f5f7fb !important; }
    .card span { color:#aab5c7 !important; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1rem; }
    .metric-label { color:var(--muted); font-size:.8rem; }
    .metric-value { color:var(--text); font-size:1.35rem; font-weight:700; margin-top:.35rem; }
    .status { color:var(--green); font-weight:600; }
    .stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid var(--line); }
    .stTabs [data-baseweb="tab"] { color:var(--muted); padding:1rem 1.1rem; font-weight:600; }
    .stTabs [aria-selected="true"] { color:var(--blue2) !important; border-bottom:2px solid var(--blue2); }
    [data-testid="stRadio"] > div { gap:.15rem; border-bottom:1px solid var(--line); overflow-x:auto; }
    [data-testid="stRadio"] label { color:var(--muted) !important; padding:.8rem 1rem; font-weight:600; white-space:nowrap; border-bottom:2px solid transparent; }
    [data-testid="stRadio"] label:has(input:checked) { color:var(--blue2) !important; border-bottom-color:var(--blue2); }
    [data-testid="stRadio"] label > div:first-child { display:none; }
    .stButton > button { background:transparent; color:var(--text); border:1px solid #40516b; border-radius:7px; font-weight:600; }
    .stButton > button:hover { color:white; border-color:var(--blue2); background:#203653; }
    button[kind="primary"] { background:var(--blue) !important; border-color:var(--blue) !important; }
    [data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stTextInput"] input { background:var(--panel2); border-color:var(--line); color:var(--text); }
    [data-baseweb="input"] input, [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; caret-color:#19a9e8; }
    [data-testid="stNumberInput"] input { color:#17304d !important; -webkit-text-fill-color:#17304d !important; caret-color:#078fd0; }
    [data-baseweb="input"] input::placeholder, [data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { color:#b7c2d2 !important; opacity:1 !important; -webkit-text-fill-color:#b7c2d2 !important; }
    [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background:#1e2d45 !important; border-color:#19a9e8 !important; }
    [data-baseweb="select"] span, [data-baseweb="select"] input { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small { color:#17304d !important; -webkit-text-fill-color:#17304d !important; }
    [data-testid="stDownloadButton"] button { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; background:#078fd0 !important; border-color:#078fd0 !important; }
    [data-testid="stChatMessage"] { background:var(--panel); border:1px solid var(--line); }
    .stAlert { background:var(--panel2); border-color:#40516b; }
    .stButton > button[type="submit"], button[kind="primary"] { background:#d8ebdf !important; border:1px solid rgba(31,35,35,0.08) !important; color:#171b1d !important; }
    .stButton > button[type="submit"]:hover, button[kind="primary"]:hover { background:#cfe4d7 !important; }
    .stButton > button[type="submit"] span, button[kind="primary"] span { color:#171b1d !important; }
    .assistant-hero { margin: 0 0 1.5rem; }
    .assistant-hero h1 { font-size: clamp(2.3rem, 4vw, 4rem) !important; letter-spacing:-0.06em; margin: 0 0 1.5rem; font-weight: 700; color: #171b1d !important; }
    .assistant-hero h1, h3, .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 { color: #171b1d !important; }
    .stMarkdown h3 { color: #171b1d !important; }
    .home-hero { min-height: 380px; padding: 3rem; margin: .5rem 0 1.5rem; border-radius: 18px; background: linear-gradient(100deg, rgba(10,31,37,.92), rgba(10,31,37,.46)), url('https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1800&q=85') center/cover; display:flex; align-items:flex-end; }
    .home-hero h1, .home-hero h1 * { color:#ffffff !important; -webkit-text-fill-color:#ffffff !important; text-shadow:0 2px 12px rgba(0,0,0,.45); font-size:clamp(2.5rem, 5vw, 5.2rem) !important; line-height:1.02; max-width:760px; margin:0; letter-spacing:-.055em; }
    .home-hero p, .home-hero p * { color:#f2faf7 !important; -webkit-text-fill-color:#f2faf7 !important; max-width:620px; font-size:1.08rem; margin:.9rem 0 0; text-shadow:0 1px 7px rgba(0,0,0,.35); }
    .home-eyebrow, .home-eyebrow * { color:#b8f5d6 !important; -webkit-text-fill-color:#b8f5d6 !important; text-transform:uppercase; letter-spacing:.14em; font-size:.75rem; font-weight:700; }
    .home-panel { background:#e3ebe6; border:1px solid rgba(31,35,35,.08); border-radius:12px; padding:1.15rem; min-height:126px; }
    .home-panel h3 { margin:.25rem 0 .45rem; font-size:1.05rem !important; }
    .home-panel p { color:#52615d !important; margin:0; font-size:.9rem; }
    .home-feature-button button { min-height:126px !important; width:100% !important; text-align:left !important; padding:1.15rem !important; background:#e3ebe6 !important; border:1px solid rgba(31,35,35,.08) !important; border-radius:12px !important; color:#173b3b !important; white-space:normal !important; }
    .home-feature-button button:hover { background:#d5e5dc !important; border-color:#78aa94 !important; }
    .home-feature-button button p { white-space:normal !important; }
    .home-stat { border-top:2px solid #8ebbaa; padding-top:.75rem; }
    .home-stat strong { display:block; font-size:1.45rem; color:#173b3b; }
    .home-stat span { color:#62716c; font-size:.8rem; }
    .card { background: rgba(209, 220, 214, 0.7) !important; border: 1px solid rgba(31,35,35,0.08) !important; }
    .card h3, .card p, .card b, .card span { color: #171b1d !important; }
    div[data-testid="stFormSubmitButton"] button {
        background: rgba(27, 33, 32, 0.18) !important; border: none !important; border-radius: 50% !important; color: #1d2424 !important; min-width: 3.25rem !important; width: 3.25rem !important; height: 3.25rem !important; padding: 0 !important; font-size: 1.7rem !important; line-height: 1 !important; display:flex; align-items:center; justify-content:center; box-shadow:none !important; }
    div[data-testid="stFormSubmitButton"] button:hover { background: rgba(27, 33, 32, 0.24) !important; }
    div[data-testid="stFormSubmitButton"] button span { color: #1d2424 !important; }
    div[data-testid="stForm"] > div {
        display:flex; align-items:center; gap:0.8rem; background:#dce7e3; border-radius: 22px; padding: 0.45rem 0.55rem 0.45rem 1rem; border: 1px solid rgba(42,58,52,0.08); box-shadow: 0 2px 0 rgba(17,24,39,0.02); }
    div[data-testid="stForm"] input {
        background: transparent !important; border: none !important; box-shadow: none !important; color: #000000 !important; -webkit-text-fill-color: #000000 !important; font-size: 1.1rem !important; padding: 0.75rem 0.5rem 0.75rem 0 !important; height: 3.2rem !important; }
    div[data-testid="stForm"] input::placeholder { color: #000000 !important; opacity: 1 !important; -webkit-text-fill-color: #000000 !important; }
    div[data-testid="stForm"] div[role="textbox"] { color: #000000 !important; }
    div[data-testid="stForm"] input, div[data-testid="stForm"] input:focus, div[data-testid="stForm"] input:active { color: #000000 !important; -webkit-text-fill-color: #000000 !important; }
    [data-testid="stTextInput"] input, [data-testid="stTextInput"] textarea { color: #000000 !important; -webkit-text-fill-color: #000000 !important; }
    [data-testid="stTextInput"] input::placeholder, [data-testid="stTextInput"] textarea::placeholder { color: #000000 !important; opacity: 1 !important; -webkit-text-fill-color: #000000 !important; }
    @media (max-width: 768px) { div[data-testid="stForm"] > div { padding-left: 0.75rem; } }
    </style>
    """, unsafe_allow_html=True)


def sidebar_controls() -> tuple[str, int, int, str, list[str], str, int, str]:
    st.sidebar.markdown("<div class='brand'><span class='brand-mark'>⚙️</span><span class='brand-title' style='font-size:1.1rem'>Trip Preferences</span></div>", unsafe_allow_html=True)
    language = st.sidebar.selectbox("Select Language / زبان منتخب کریں", ["English", "Roman Urdu", "Urdu"])
    st.sidebar.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    budget = st.sidebar.slider("Budget (PKR)", min_value=10000, max_value=500000, value=100000, step=5000)
    duration = st.sidebar.number_input("Duration (Days)", min_value=1, max_value=30, value=5)
    origin = st.sidebar.selectbox("Starting point", ["Current location"] + list(DESTINATION_COORDINATES), key="sidebar_origin")
    travelers = st.sidebar.number_input("Travellers", min_value=1, max_value=20, value=1, step=1, key="sidebar_travelers")
    trip_type = st.sidebar.selectbox("Trip type", ["One-way", "Return trip"], key="sidebar_trip_type")
    style = st.sidebar.selectbox("Travel Style", ["Auto", "Budget Friendly", "Comfort", "Luxury", "Family Friendly", "Adventure"])
    interests = st.sidebar.multiselect("Interests & Activities", ["Adventure", "Mountains", "Nature", "Family", "Food", "Culture", "History", "Photography", "Lakes"], default=[])
    st.sidebar.markdown("<div class='rule'></div><div class='card'><b>💡 Plan with confidence</b><br><span style='color:#aab5c7'>Select your preferences, then ask the AI Travel Assistant for recommendations.</span></div>", unsafe_allow_html=True)
    return language, budget, duration, style, [item.lower() for item in interests], origin, int(travelers), trip_type


inject_theme()
initialize_database()
destinations = destination_rows()
train_lightweight_ranker(destinations)
for key, default in {"chat_history": [], "extracted_entities": {}, "recommendations": [], "current_itinerary": "", "uploaded_image_hash": None, "landmark_result": None, "weather_result": ""}.items():
    st.session_state.setdefault(key, default)

language, budget_filter, duration_filter, style_filter, interests_filter, origin_filter, travelers_filter, trip_type_filter = sidebar_controls()
st.markdown("<div class='brand'><span class='brand-mark'>✈️</span><span class='brand-title'>AI Travel Agent</span></div><div class='subtitle'>Your intelligent multilingual travel planning assistant</div>", unsafe_allow_html=True)

tab_home, tab_assistant, tab_travel_os, tab_weather, tab_landmark, tab_status = st.tabs(["⌂ Home", "🗺️ AI Trip Planner", "🧭 Travel OS", "🌤️ Live Weather", "📷 Landmark Recognition", "💻 System Status"])

with tab_home:
    st.markdown("""
    <section class="home-hero">
      <div>
        <div class="home-eyebrow">AI Travel Agent · intelligent journeys</div>
        <h1>Go further.<br>Travel smarter.</h1>
        <p>Plan meaningful trips with one calm workspace for destinations, budgets, stays, routes, local insight and every detail in between.</p>
      </div>
    </section>
    """, unsafe_allow_html=True)
    home_destination = st.text_input("Where are you going?", value=st.session_state.extracted_entities.get("destination") or "Hunza", key="home_destination", placeholder="Search a destination")
    home_cols = st.columns([1.2, 1, 1, 1])
    with home_cols[0]:
        if st.button("Start planning", type="primary", use_container_width=True, key="home_start"):
            st.session_state.extracted_entities = {"destination": home_destination, "budget": budget_filter, "duration": duration_filter, "travelers": travelers_filter, "origin": origin_filter, "trip_type": trip_type_filter, "interests": interests_filter, "travel_style": style_filter.lower(), "missing": []}
            st.toast("Trip workspace ready")
    with home_cols[1]:
        st.markdown("<div class='home-stat'><strong>36</strong><span>Travel capabilities</span></div>", unsafe_allow_html=True)
    with home_cols[2]:
        st.markdown(f"<div class='home-stat'><strong>{len(destinations)}</strong><span>Curated destinations</span></div>", unsafe_allow_html=True)
    with home_cols[3]:
        st.markdown(f"<div class='home-stat'><strong>{len(favorite_rows())}</strong><span>Saved trips</span></div>", unsafe_allow_html=True)
    st.markdown("### Everything for the journey")
    feature_columns = st.columns(4)
    feature_cards = [("01 · Discover", "Find destinations, attractions, food and local guidance.", "Discover"), ("02 · Plan", "Build weather-aware days, routes, hotels and packing lists.", "Plan"), ("03 · Manage", "Track spending, documents, reminders and trip progress.", "My Trip"), ("04 · Explore", "Translate signs, recognize landmarks and stay safety-aware.", "Language")]
    for column, (title, description, target) in zip(feature_columns, feature_cards):
        with column:
            st.markdown("<div class='home-feature-button'>", unsafe_allow_html=True)
            if st.button(f"{title}\n\n{description}", key=f"home_feature_{title}", use_container_width=True):
                st.session_state.home_feature = title
                st.session_state.os_tools = target
                st.session_state.home_target = target
                st.session_state.active_section = target
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Included travel tools")
    tool_mapping = {
        "AI Itinerary Planner": "AI Trip Planner",
        "Interactive Budget Dashboard": "Money",
        "Smart Packing Checklist": "Plan",
        "Group Expense Splitter": "Money",
        "Interactive Map View": "Discover",
        "Weather Widget": "Live Weather",
        "Multi-Page Navigation": "Travel OS",
        "Report Export (Excel/PDF)": "Trip Dashboard",
        "Currency Converter": "Money",
        "SQLite Database Integration": "System Status",
    }
    tool_columns = st.columns(5)
    for idx, (tool_name, handled_by) in enumerate(tool_mapping.items()):
        with tool_columns[idx % 5]:
            st.markdown("<div class='home-feature-button'>", unsafe_allow_html=True)
            if st.button(f"{tool_name}", key=f"tool_{tool_name}", use_container_width=True):
                st.session_state.home_feature = tool_name
                st.session_state.home_target = handled_by
                st.session_state.active_section = handled_by
                if handled_by == "AI Trip Planner":
                    st.session_state.home_feature = "AI Itinerary Planner"
                    st.session_state.home_target = "AI Trip Planner"
                elif handled_by == "Live Weather":
                    st.session_state.home_feature = "Weather Widget"
                    st.session_state.home_target = "Weather Widget"
                elif handled_by == "Travel OS":
                    st.session_state.home_feature = "Multi-Page Navigation"
                    st.session_state.home_target = "Travel OS"
                elif handled_by == "System Status":
                    st.session_state.home_feature = "SQLite Database Integration"
                    st.session_state.home_target = "System Status"
                else:
                    st.session_state.home_feature = tool_name
                    st.session_state.home_target = tool_name
            st.markdown("</div>", unsafe_allow_html=True)

    selected_feature = st.session_state.get("home_feature")
    if selected_feature:
        feature_destination = st.session_state.get("home_destination", "Hunza")
        feature_target = st.session_state.get("home_target") or next((item[2] for item in feature_cards if item[0] == selected_feature), tool_mapping.get(selected_feature, "Discover"))
        st.success(f"{selected_feature} opened for {feature_destination}. The app is now focused on the {feature_target} section.")

        if feature_target in {"Discover", "Money", "Plan", "Language", "Safety", "My Trip"}:
            st.session_state.os_tools = feature_target
            st.info(f"Opening Travel OS → {feature_target}.")
        elif feature_target == "AI Trip Planner":
            st.info("Opening AI Trip Planner.")
        elif feature_target == "Weather Widget":
            st.info("Opening Live Weather.")
        elif feature_target == "System Status":
            st.info("Opening System Status.")

    st.markdown("### Built around your trip")
    home_left, home_right = st.columns([1.1, 1])
    with home_left:
        st.markdown("**A practical travel companion, from the first idea to the last day.**")
        st.caption("Use the AI Trip Planner for a complete itinerary, or open Travel OS for focused tools while you prepare and travel.")
    with home_right:
        st.info("Tip: choose your budget, duration and interests in the sidebar before starting a plan.")

with tab_assistant:
    st.markdown("<div class='assistant-hero'><h1>Tell Ayla what you have in mind</h1></div>", unsafe_allow_html=True)
    left, right = st.columns([1.25, .9])
    with left:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        with st.form(key="assistant_prompt_form", clear_on_submit=True):
            prompt_col, submit_col = st.columns([12, 1.3])
            with prompt_col:
                prompt = st.text_input(
                    "Trip idea",
                    key="assistant_prompt_input",
                    label_visibility="collapsed",
                    placeholder="e.g. 4 din Hunza trip, 100k budget, photography ar nature",
                )
            with submit_col:
                submitted = st.form_submit_button("↑", help="Send prompt", use_container_width=True)

        if submitted and prompt.strip():
            st.session_state.chat_history.append({"role": "user", "content": prompt.strip()})
            entities = extract_entities(prompt.strip(), language)
            entities["budget"] = entities["budget"] or budget_filter
            entities["duration"] = entities["duration"] or duration_filter
            entities["travel_style"] = entities["travel_style"] or style_filter.lower()
            entities["interests"] = sorted(set(entities["interests"] + interests_filter))
            entities["origin"] = entities.get("origin") or origin_filter
            entities["travelers"] = entities.get("travelers") or travelers_filter
            entities["trip_type"] = trip_type_filter
            st.session_state.extracted_entities = entities
            intent = classify_intent(prompt.strip())
            missing = entities["missing"]
            answer = f"I understood this as **{intent.replace('_', ' ')}**. Please confirm: {', '.join(missing)}." if missing else "I extracted your trip details. Confirm them on the right to get recommendations."
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()
    with right:
        st.markdown("### 🔑 Confirm your trip")
        entities = st.session_state.extracted_entities
        if not entities:
            st.info("Your destination, budget, and duration will appear here.")
        else:
            destination_query = st.text_input("Search or type any destination", value=entities.get("destination") or "", placeholder="Murree, Antalya, Hunza...")
            names = [row["name"] for row in destinations]
            matches = [name for name in names if destination_query.lower().strip() in name.lower()] if destination_query.strip() else names
            selected = st.selectbox("Matching places", ["Use typed destination"] + matches) if matches else "Use typed destination"
            selected_destination = destination_query.strip() if selected == "Use typed destination" else selected
            confirmed_budget = st.number_input("Total budget (PKR)", min_value=0, value=int(entities.get("budget") or budget_filter), step=5000, key="confirmed_budget")
            confirmed_duration = st.number_input("Days", min_value=1, max_value=30, value=int(entities.get("duration") or duration_filter), key="confirmed_duration")
            confirmed_travelers = st.number_input("Travellers", min_value=1, max_value=20, value=int(entities.get("travelers") or travelers_filter), key="confirmed_travelers")
            confirmed_origin = st.selectbox("Starting point", ["Current location"] + list(DESTINATION_COORDINATES), index=(["Current location"] + list(DESTINATION_COORDINATES)).index(entities.get("origin")) if entities.get("origin") in (["Current location"] + list(DESTINATION_COORDINATES)) else 0, key="confirmed_origin")
            confirmed_trip_type = st.selectbox("Trip type", ["One-way", "Return trip"], index=1 if entities.get("trip_type") == "Return trip" else 0, key="confirmed_trip_type")
            if st.button("🔎 Get Recommendations", type="primary", use_container_width=True):
                if not selected_destination:
                    st.error("Please type a destination first.")
                else:
                    entities.update({"destination": selected_destination, "budget": confirmed_budget, "duration": confirmed_duration, "travelers": confirmed_travelers, "origin": confirmed_origin, "trip_type": confirmed_trip_type})
                    st.session_state.extracted_entities = entities
                    recommendation_data = destinations
                    if selected_destination.lower() not in {row["name"].lower() for row in destinations}:
                        recommendation_data = [{"name": selected_destination, "province_country": "User-selected destination", "tags": "travel", "avg_daily_cost": max(1, confirmed_budget // max(confirmed_duration, 1)), "rating": 0.0, "description": "No verified local description is available yet."}] + destinations
                    st.session_state.recommendations = recommend_destinations(entities, recommendation_data)
                    st.rerun()
    if st.session_state.recommendations:
        st.markdown("### 🧠 Smart Recommendations")
        cards = st.columns(len(st.session_state.recommendations))
        for card, item in zip(cards, st.session_state.recommendations):
            with card:
                st.markdown(f"<div class='card'><h3>{item['name']}</h3><p style='color:#aab5c7'>{item['description']}</p><b>Fit: {item['score']:.0%}</b><br><span style='color:#aab5c7'>Estimate: PKR {item['estimated_cost']:,}</span></div>", unsafe_allow_html=True)
        if st.button("🗺️ Generate complete trip plan", type="primary"):
            entities = st.session_state.extracted_entities
            context = format_context(retrieve(f"{entities['destination']} {' '.join(entities.get('interests', []))}"))
            weather = get_weather(entities["destination"])
            itinerary = full_trip_plan(entities, generate_itinerary(entities, context, weather), weather)
            st.session_state.current_itinerary = itinerary
            log_trip(entities["destination"], entities["budget"], entities["duration"], itinerary)
            st.rerun()
    if st.session_state.current_itinerary:
        st.markdown(st.session_state.current_itinerary)

    st.markdown("---")
    st.subheader("🧰 Planner Tools")
    planner_tool_tabs = st.tabs(["💰 Budget, Food & Packing", "🧭 Hotels, Transport & Places"])
    planner_entities = st.session_state.extracted_entities
    if not planner_entities.get("destination"):
        st.info("Enter a destination above and generate recommendations to unlock all planner tools.")
    else:
        with planner_tool_tabs[0]:
            planner_left, planner_right = st.columns(2)
            with planner_left:
                st.markdown("#### 💰 Budget Breakdown")
                planner_breakdown = budget_breakdown(int(planner_entities.get("budget", budget_filter)), int(planner_entities.get("duration", duration_filter)))
                for category, amount in planner_breakdown.items():
                    st.markdown(f"<div class='metric'><span>{category}</span><span style='float:right;font-weight:700'>PKR {amount:,}</span></div>", unsafe_allow_html=True)
            with planner_right:
                st.markdown("#### 🧳 Packing Checklist")
                for item in packing_list(planner_entities):
                    st.checkbox(item, value=False, key=f"planner_pack_{item}")
            st.markdown("#### 🍽️ Food Recommendations")
            food_columns = st.columns(2)
            for index, food in enumerate(food_recommendations(planner_entities["destination"], planner_entities.get("interests", []))):
                food_columns[index % 2].markdown(f"- {food}")

        with planner_tool_tabs[1]:
            planner_hub_tool = st.radio("Choose planner tool", ["🏨 Hotels", "🚗 Transport", "📍 Nearby Places", "📄 Travel Guide", "📊 Trip Dashboard"], horizontal=True, key="planner_hub_tool")
            if planner_hub_tool == "🏨 Hotels":
                planner_budget = int(planner_entities.get("budget", 0))
                planner_duration = int(planner_entities.get("duration", 1))
                hotel_results = recommend_hotels(planner_entities["destination"], load_hotel_directory(), hotel_budget_per_night(planner_budget, planner_duration) or None, 3)
                hotel_columns = st.columns(min(3, len(hotel_results))) if hotel_results else []
                for index, hotel in enumerate(hotel_results):
                    with hotel_columns[index % len(hotel_columns)]:
                        st.markdown(f"<div class='card'><h3>{hotel['name']}</h3><b>PKR {float(hotel['price_per_night']):,.0f}/night</b><br>⭐ {float(hotel['guest_rating']):.1f}/5 · 📞 {hotel['contact_number']}<br>📍 {hotel['address']}</div>", unsafe_allow_html=True)
            elif planner_hub_tool == "🚗 Transport":
                for option in transport_options("Current location", planner_entities["destination"], 1, "One-way", int(planner_entities.get("budget", 0))):
                    st.markdown(f"<div class='metric'><b>{option['mode']}</b><br>PKR {option['cost']:,} · {option['distance_km']} km · {option['hours']:.1f} hours</div>", unsafe_allow_html=True)
            elif planner_hub_tool == "📍 Nearby Places":
                place_details = nearby_place_details(planner_entities["destination"])
                place_names = [place["name"] for place in place_details]
                selected_name = st.session_state.get("planner_selected_place", place_names[0])
                if selected_name not in place_names:
                    selected_name = place_names[0]
                place_columns = st.columns(min(2, len(place_details)))
                for index, place in enumerate(place_details):
                    with place_columns[index % len(place_columns)]:
                        if st.button(f"📍 {place['name']}", key=f"planner_nearby_{planner_entities['destination']}_{index}", use_container_width=True):
                            st.session_state.planner_selected_place = place["name"]
                            st.rerun()
                        st.caption(place["visit"])
                selected_place = next(place for place in place_details if place["name"] == selected_name)
                st.markdown(f"### 📷 {selected_place['name']}")
                detail_left, detail_right = st.columns([1.2, 1])
                with detail_left:
                    st.image(selected_place["image"], use_container_width=True)
                with detail_right:
                    st.markdown(f"**{selected_place['description']}**")
                    st.markdown(f"📌 **Visit plan:** {selected_place['visit']}")
                    st.markdown(f"💡 **Travel tip:** {selected_place['tip']}")
                    st.caption("Confirm current access, timings, weather and local conditions before visiting.")
            elif planner_hub_tool == "📄 Travel Guide":
                st.markdown(travel_guide(planner_entities["destination"], planner_entities))
            elif planner_hub_tool == "📊 Trip Dashboard":
                planner_trips = favorite_rows()
                st.metric("Saved trips", len(planner_trips))
                st.metric("Planned budget", f"PKR {sum(float(trip['total_budget']) for trip in planner_trips):,.0f}")

with tab_travel_os:
    st.subheader("🧭 Travel OS")
    st.caption("One workspace for planning, discovery, safety, documents and on-trip control.")
    os_entities = st.session_state.extracted_entities
    os_destination = st.text_input("Trip destination", value=os_entities.get("destination") or "Lahore", key="os_destination")
    os_tools = st.radio("Open workspace", ["Discover", "Money", "Plan", "Language", "Safety", "My Trip"], horizontal=True, key="os_tools")

    if os_tools == "Discover":
        st.markdown("### Smart map and discovery")
        map_locations = [{"name": name, "lat": coords[0], "lon": coords[1]} for name, coords in DESTINATION_COORDINATES.items()]
        map_col, guide_col = st.columns([1.15, 1])
        with map_col:
            selected_map = st.selectbox("Map destination", list(DESTINATION_COORDINATES), index=list(DESTINATION_COORDINATES).index(os_destination) if os_destination in DESTINATION_COORDINATES else 0, key="os_map_destination")
            lat, lon = DESTINATION_COORDINATES[selected_map]
            st.map([{"lat": lat, "lon": lon}], latitude="lat", longitude="lon", size=80, color="#0e7490", zoom=6)
            st.caption(f"Selected location: {selected_map} · Latitude {lat}, Longitude {lon}")
        with guide_col:
            st.markdown(f"### AI local guide: {selected_map}")
            st.markdown(travel_guide(selected_map, {**os_entities, "destination": selected_map}))
            st.markdown("#### Attractions and activities")
            for place in nearby_place_details(selected_map):
                place_col, save_col = st.columns([4, 1])
                place_col.write(f"**{place['name']}** · {place['visit']}")
                if save_col.button("Save", key=f"os_save_{selected_map}_{place['name']}"):
                    save_place(selected_map, place["name"], "attraction", place["description"])
                    st.success("Saved")
        st.markdown("#### Destination discovery")
        discovery = recommend_destinations({**os_entities, "destination": os_destination}, destinations, limit=4)
        st.dataframe([{"Destination": item["name"], "Match": f"{item['score']:.0%}", "Estimated PKR": item["estimated_cost"], "Why": item["description"]} for item in discovery], use_container_width=True, hide_index=True)

    elif os_tools == "Money":
        st.markdown("### Live currency and spending control")
        money_left, money_right = st.columns(2)
        with money_left:
            amount = st.number_input("Amount", min_value=0.0, value=10000.0, step=500.0, key="os_amount")
            source = st.selectbox("From", ["PKR", "USD", "EUR", "GBP", "AED", "TRY"], key="os_source")
            target = st.selectbox("To", ["PKR", "USD", "EUR", "GBP", "AED", "TRY"], index=1, key="os_target")
            st.metric("Converted amount", f"{convert_currency(amount, source, target):,.2f} {target}")
            st.caption("Indicative in-app rates. Confirm the live rate with your bank or exchange counter.")
        with money_right:
            st.markdown("#### Add expense")
            with st.form("os_expense_form"):
                expense_category = st.selectbox("Category", ["Stay", "Transport", "Food", "Activities", "Shopping", "Emergency"])
                expense_description = st.text_input("Description")
                expense_amount = st.number_input("Amount (PKR)", min_value=0.0, step=100.0)
                expense_paid_by = st.text_input("Paid by", value="Me")
                if st.form_submit_button("Add expense", type="primary") and expense_amount > 0:
                    add_expense(os_destination, expense_category, expense_description, expense_amount, expense_paid_by)
                    st.success("Expense added")
        expenses = expense_rows(os_destination)
        total_spent = sum(float(row["amount"]) for row in expenses)
        planned = int(os_entities.get("budget") or budget_filter)
        budget_col, spent_col, remain_col = st.columns(3)
        budget_col.metric("Planned budget", f"PKR {planned:,}")
        spent_col.metric("Spent", f"PKR {total_spent:,.0f}")
        remain_col.metric("Remaining", f"PKR {planned - total_spent:,.0f}")
        if expenses:
            st.dataframe([{"Category": row["category"], "Description": row["description"], "PKR": row["amount"], "Paid by": row["paid_by"]} for row in expenses], use_container_width=True, hide_index=True)
        st.markdown("#### Group expense splitter")
        group_total = st.number_input("Group expense total (PKR)", min_value=0.0, value=float(total_spent), step=500.0, key="group_total")
        group_size = st.number_input("People sharing", min_value=1, max_value=30, value=2, key="group_size")
        st.info(f"Equal share: PKR {group_total / group_size:,.0f} per person")

    elif os_tools == "Plan":
        st.markdown("### AI smart packing and weather-aware daily planner")
        plan_left, plan_right = st.columns(2)
        with plan_left:
            weather = get_weather(os_destination)
            st.metric("Current weather", weather)
            st.markdown("#### Adaptive packing list")
            packing = packing_list({**os_entities, "destination": os_destination})
            for item in packing:
                st.checkbox(item, value=False, key=f"os_pack_{os_destination}_{item}")
        with plan_right:
            st.markdown("#### Daily travel planner")
            plan_days = st.number_input("Days to schedule", 1, 30, int(os_entities.get("duration") or duration_filter), key="os_plan_days")
            for day in range(1, int(plan_days) + 1):
                places = nearby_place_details(os_destination)
                place = places[(day - 1) % len(places)]
                st.markdown(f"**Day {day} · {place['name']}**  \nMorning route, weather buffer, local meal and safe return route.")
        st.markdown("#### Route and transport planner")
        route_origin = st.selectbox("Route origin", ["Current location"] + list(DESTINATION_COORDINATES), key="os_route_origin")
        route_budget = int(os_entities.get("budget") or budget_filter)
        route_destination = os_destination if os_destination in DESTINATION_COORDINATES else list(DESTINATION_COORDINATES)[0]
        route_options = transport_options(route_origin, route_destination, 1, "One-way", route_budget)
        st.dataframe([{"Mode": item["mode"], "Distance km": item["distance_km"], "Hours": round(item["hours"], 1), "Estimated PKR": item["cost"]} for item in route_options], use_container_width=True, hide_index=True)

    elif os_tools == "Language":
        st.markdown("### AI travel translator")
        language_target = st.selectbox("Local language", ["Urdu", "Arabic", "Turkish", "English"], key="os_language_target")
        phrase_cols = st.columns(2)
        for index, (english, local) in enumerate(local_phrases(language_target)):
            phrase_cols[index % 2].markdown(f"**{english}**  \n{local}")
        st.markdown("#### Conversation translator")
        translate_text = st.text_area("Text to translate", placeholder="Type a phrase for a hotel, taxi or restaurant...", key="os_translate_text")
        if st.button("Translate phrase", type="primary", key="os_translate"):
            translated_phrase, translation_note = translate_phrase(translate_text, language_target)
            result_language = "English" if is_roman_urdu(translate_text) else language_target
            if translated_phrase:
                st.success(f"{result_language} translation: {translated_phrase}")
            else:
                st.warning(translation_note)
            if translated_phrase:
                st.caption(translation_note)
        st.markdown("#### Visual / sign and menu translator")
        sign_image = st.file_uploader("Upload a sign or menu photo", type=["jpg", "jpeg", "png"], key="os_sign_image")
        if sign_image:
            from PIL import Image
            uploaded_image = Image.open(sign_image)
            preview_col, info_col = st.columns([1.15, 1])
            with preview_col:
                st.image(uploaded_image, caption=f"Selected image: {sign_image.name}", use_container_width=True)
            with info_col:
                st.markdown("**Selected image**")
                st.caption(f"{sign_image.name} · {uploaded_image.width} × {uploaded_image.height}px")
                st.caption("This is the image that will be analyzed when you click Read image.")
        if sign_image and st.button("Read image", key="os_read_sign"):
            image_bytes = sign_image.getvalue()
            st.session_state.os_image_details = describe_image(image_bytes, get_secret("OPENAI_API_KEY"), get_secret("OPENAI_MODEL"))
            st.session_state.os_image_category = recognize_landmark(image_bytes, get_secret("OPENAI_API_KEY"), get_secret("OPENAI_MODEL"))
        image_details = st.session_state.get("os_image_details")
        if image_details:
            if "summary" in image_details:
                st.success(image_details["summary"])
                detail_rows = [
                    ("Objects visible", image_details["objects"]),
                    ("Food / items", image_details["food_or_items"]),
                    ("Setting", image_details["setting"]),
                    ("Readable text", image_details["visible_text"]),
                    ("Travel context", image_details["travel_context"]),
                ]
                for label, value in detail_rows:
                    st.markdown(f"**{label}:** {value}")
            else:
                st.warning(image_details.get("description", "Detailed image information is unavailable."))
                st.caption(image_details.get("message", ""))

    elif os_tools == "Safety":
        st.markdown(f"### Tourist safety center · {os_destination}")
        safety_cols = st.columns(2)
        with safety_cols[0]:
            st.markdown("#### Scam and safety advisor")
            for advice in safety_brief(os_destination):
                st.markdown(f"- {advice}")
        with safety_cols[1]:
            st.markdown("#### Emergency assistance")
            st.error("Emergency: call your local police, ambulance or rescue service immediately. This app cannot dispatch emergency services.")
            st.dataframe([{"Service": "Pakistan Police", "Number": "15"}, {"Service": "Rescue / Ambulance", "Number": "1122"}, {"Service": "Tourist contact", "Number": "Verify with destination authority"}], use_container_width=True, hide_index=True)
            st.markdown("#### Emergency contact directory")
            st.text_input("Trusted contact name", key="os_trusted_name")
            st.text_input("Trusted contact phone", key="os_trusted_phone")

    else:
        st.markdown("### My trip control center")
        trip_col, doc_col = st.columns(2)
        with trip_col:
            st.markdown("#### Trip progress and checklist")
            items = packing_list({**os_entities, "destination": os_destination}) + ["Confirm hotel", "Download offline guide", "Share emergency plan"]
            for item in items:
                save_checklist_item(os_destination, item)
            rows = checklist_rows(os_destination)
            completed = 0
            for row in rows:
                checked = st.checkbox(row["item"], value=bool(row["completed"]), key=f"os_check_{row['id']}")
                if checked:
                    completed += 1
                if checked != bool(row["completed"]):
                    set_checklist_item(row["id"], checked)
            st.progress(completed / max(len(rows), 1), text=f"Trip progress: {completed}/{len(rows)} tasks")
            st.markdown("#### Activity reminder")
            with st.form("os_reminder_form"):
                reminder_title = st.text_input("Reminder", value="Check tomorrow's route")
                reminder_date = st.date_input("Date")
                reminder_time = st.time_input("Time")
                if st.form_submit_button("Add reminder"):
                    add_reminder(os_destination, reminder_title, reminder_date.isoformat(), reminder_time.strftime("%H:%M"))
                    st.success("Reminder saved")
            for row in reminder_rows(os_destination):
                st.caption(f"{row['reminder_date']} {row['reminder_time']} · {row['title']}")
        with doc_col:
            st.markdown("#### Travel document organizer")
            with st.form("os_document_form"):
                document_name = st.text_input("Document name", placeholder="Passport / visa / insurance")
                document_type = st.selectbox("Type", ["Passport", "Visa", "Insurance", "Booking", "Other"])
                expiry_date = st.date_input("Expiry or travel date")
                document_notes = st.text_input("Notes")
                if st.form_submit_button("Add document") and document_name.strip():
                    add_document(os_destination, document_name.strip(), document_type, expiry_date.isoformat(), document_notes)
                    st.success("Document record saved")
            docs = document_rows(os_destination)
            if docs:
                st.dataframe([{"Document": row["document_name"], "Type": row["document_type"], "Date": row["expiry_date"], "Notes": row["notes"]} for row in docs], use_container_width=True, hide_index=True)
            saved = saved_place_rows(os_destination)
            st.markdown(f"#### Saved places ({len(saved)})")
            for place in saved:
                st.caption(f"{place['place_name']} · {place['place_type']}")
        st.markdown("#### Offline travel guide")
        offline_guide = travel_guide(os_destination, {**os_entities, "destination": os_destination})
        st.download_button("Download offline guide", data=offline_guide, file_name=f"{os_destination.replace(' ', '_')}_offline_guide.md", mime="text/markdown", key="os_offline_guide")

_REMOVED_LEGACY_TABS = """Legacy top-level Budget/Food/Packing and Hotels/Transport/Places tabs removed.
Their functionality is available inside AI Trip Planner > Planner Tools.
with tab_tools:
    st.subheader("💰 Budget, Food & Packing")
    entities = st.session_state.extracted_entities
    if not entities.get("destination"):
        st.info("First create a trip in AI Travel Assistant, then use these tools for its budget, packing and food plan.")
    else:
        tools_left, tools_right = st.columns(2)
        with tools_left:
            st.markdown("### 💰 Smart Budget Breakdown")
            breakdown = budget_breakdown(int(entities.get("budget", budget_filter)), int(entities.get("duration", duration_filter)))
            for category, amount in breakdown.items():
                st.markdown(f"<div class='metric'><span>{category}</span><span style='float:right;font-weight:700'>PKR {amount:,}</span></div>", unsafe_allow_html=True)
            st.markdown("### 🧠 AI Trip Optimization")
            optimization = st.selectbox("Optimize your trip for", ["Best budget fit", "More activities", "Relaxed pace", "Family comfort"], key="optimization_mode")
            if st.button("⚡ Optimize trip", type="primary", key="optimize_trip"):
                optimized = dict(entities)
                if optimization == "More activities":
                    optimized["interests"] = sorted(set(optimized.get("interests", []) + ["adventure"]))
                if optimization == "Relaxed pace":
                    optimized["duration"] = min(30, int(optimized.get("duration", duration_filter)) + 1)
                st.session_state.extracted_entities = optimized
                st.session_state.recommendations = recommend_destinations(optimized, destinations)
                st.success(f"Trip optimized for {optimization.lower()}.")
        with tools_right:
            st.markdown("### 🧳 Smart Packing List")
            for item in packing_list(entities):
                st.checkbox(item, value=False, key=f"pack_{item}")
            st.markdown("### 🍽️ Food Recommendations")
            for food in food_recommendations(entities["destination"], entities.get("interests", [])):
                st.markdown(f"- {food}")
        st.markdown("### 📄 Save or Download")
        if st.session_state.current_itinerary:
            pdf_bytes = itinerary_pdf(st.session_state.current_itinerary, f"{entities['destination']} Travel Itinerary")
            download_col, save_col = st.columns(2)
            with download_col:
                st.download_button("📄 Download Itinerary as PDF", data=pdf_bytes, file_name=f"{entities['destination'].replace(' ', '_')}_itinerary.pdf", mime="application/pdf", use_container_width=True)
            with save_col:
                if st.button("❤️ Save / Favorite Trip", use_container_width=True):
                    save_favorite(entities["destination"], int(entities["budget"]), int(entities["duration"]), st.session_state.current_itinerary)
                    st.success("Trip saved to favorites.")
        else:
            st.info("Generate an itinerary first to enable PDF download and favorites.")
            if st.button("🗺️ Generate itinerary now", type="primary", key="tools_generate_itinerary"):
                context = format_context(retrieve(f"{entities['destination']} {' '.join(entities.get('interests', []))}"))
                weather = get_weather(entities["destination"])
                itinerary = full_trip_plan(entities, generate_itinerary(entities, context, weather), weather)
                st.session_state.current_itinerary = itinerary
                log_trip(entities["destination"], int(entities["budget"]), int(entities["duration"]), itinerary)
                st.rerun()
        favorites = favorite_rows()
        if favorites:
            st.markdown("### ❤️ Saved Favorite Trips")
            for favorite in favorites[:5]:
                st.markdown(f"<div class='metric'><b>{favorite['destination']}</b><br><span style='color:#aab5c7'>{favorite['duration_days']} days · PKR {favorite['total_budget']:,}</span></div>", unsafe_allow_html=True)

    with tab_hub:
        st.subheader("🧭 Hotels, Transport & Places")
        hub_entities = st.session_state.extracted_entities
        hub_destination = hub_entities.get("destination") or st.text_input("Choose a destination for travel tools", value="Lahore", key="hub_destination")
        hub_view = st.radio("Open tool", ["⚖️ Compare Destinations", "🏨 Hotel Finder", "🚗 Transport Planner", "📍 Explore Nearby", "📄 Travel Guide", "❤️ My Trips", "📊 Trip Dashboard"], horizontal=True, label_visibility="collapsed", key="hub_view")

        if hub_view == "⚖️ Compare Destinations":
            st.markdown("### ⚖️ Compare Destinations")
            names = [row["name"] for row in destinations]
            chosen = st.multiselect("Select two or more destinations", names, default=names[:3], key="compare_places")
            if len(chosen) >= 2:
                comparison = compare_destinations(chosen, int(hub_entities.get("budget", budget_filter)), int(hub_entities.get("duration", duration_filter)))
                for item in comparison:
                    st.markdown(f"<div class='card'><h3>{item['name']}</h3><span>{item['province_country']}</span><br><b>Estimated trip: PKR {item['trip_estimate']:,}</b><br><span>Budget fit: {item['budget_fit']}% · Rating: {item['rating']}/5</span></div>", unsafe_allow_html=True)
            else:
                st.info("Select at least two destinations to compare.")

        elif hub_view == "🏨 Hotel Finder":
            st.markdown("### 🏨 Nearest & Best Hotels")
            st.caption("Local planning directory. Confirm availability, price and contact details before booking.")
            hotel_destinations = list(DESTINATION_COORDINATES)
            hotel_destination = st.selectbox("Where are you staying?", hotel_destinations, index=hotel_destinations.index(hub_destination) if hub_destination in hotel_destinations else 0, key="hotel_destination")
            hotel_budget = st.number_input("Preferred rent per night (PKR)", min_value=0, value=0, step=1000, help="Use 0 to show the best options at every price.", key="hotel_budget")
            hotel_limit = st.slider("Hotels to show", min_value=1, max_value=5, value=3, key="hotel_limit")
            hotel_results = recommend_hotels(hotel_destination, load_hotel_directory(), hotel_budget or None, hotel_limit)
            hotel_columns = st.columns(min(3, len(hotel_results))) if hotel_results else []
            for index, hotel in enumerate(hotel_results):
                with hotel_columns[index % len(hotel_columns)]:
                    st.markdown(
                        f"<div class='card'><h3>{hotel['name']}</h3>"
                        f"<p>📍 {hotel['address']}<br>📞 {hotel['contact_number']}</p>"
                        f"<b>PKR {float(hotel['price_per_night']):,.0f} / night</b><br>"
                        f"⭐ {float(hotel['guest_rating']):.1f}/5 ({int(hotel['review_count'])} reviews) · {float(hotel['star_rating']):.0f}-star<br>"
                        f"🚗 {float(hotel['distance_km']):.1f} km from destination center<br>"
                        f"<span>{hotel['amenities'].replace(' | ', ' · ')}</span></div>", unsafe_allow_html=True)

        elif hub_view == "🚗 Transport Planner":
            st.markdown(f"### 🚗 Transport Planner for {hub_destination}")
            route_locations = ["Current location"] + list(DESTINATION_COORDINATES)
            route_left, route_right = st.columns(2)
            with route_left:
                origin = st.selectbox("Starting point", route_locations, index=0, key="transport_origin")
                travelers = st.number_input("Travellers", min_value=1, max_value=20, value=2, step=1, key="transport_travelers")
            with route_right:
                route_destination = st.selectbox("Destination", list(DESTINATION_COORDINATES), index=list(DESTINATION_COORDINATES).index(hub_destination) if hub_destination in DESTINATION_COORDINATES else 0, key="transport_destination")
                trip_type = st.selectbox("Trip type", ["One-way", "Return trip"], key="transport_trip_type")
            transport_budget = st.number_input("Transport budget (PKR)", min_value=0, value=0, step=1000, help="Use 0 to compare without a budget filter.", key="transport_budget")
            transport_sort = st.selectbox("Prioritize", ["Best overall", "Lowest total fare", "Fastest route", "Lowest fare per person"], key="transport_sort")
            options = transport_options(origin, route_destination, int(travelers), trip_type, int(transport_budget))
            sort_keys = {
                "Best overall": lambda item: (item["budget_fit"] if item["budget_fit"] is not None else 50, -item["cost"]),
                "Lowest total fare": lambda item: item["cost"],
                "Fastest route": lambda item: item["hours"],
                "Lowest fare per person": lambda item: item["per_person"],
            }
            options = sorted(options, key=sort_keys[transport_sort], reverse=transport_sort == "Best overall")
            if options:
                transport_columns = st.columns(min(3, len(options)))
                for index, option in enumerate(options):
                    duration = f"{option['hours']:.1f} hr" if option["hours"] < 10 else f"{option['hours']:.0f} hr"
                    budget_line = f" · Budget fit: {option['budget_fit']}%" if option["budget_fit"] is not None else ""
                    with transport_columns[index % len(transport_columns)]:
                        st.markdown(
                            f"<div class='card'><h3>{option['mode']}</h3>"
                            f"<b>PKR {option['cost']:,} total</b><br>"
                            f"PKR {option['per_person']:,} per person{budget_line}<br>"
                            f"📍 {option['distance_km']:,} km · ⏱️ {duration}<br>"
                            f"<span>Best for: {option['best_for']}<br>{option['note']}</span></div>", unsafe_allow_html=True)
                        if st.button(f"📅 Book {option['mode']} now", key=f"book_transport_{index}_{route_destination}", type="primary", use_container_width=True):
                            st.session_state.selected_transport_booking = index
                            st.rerun()
                selected_booking_index = st.session_state.get("selected_transport_booking")
                if selected_booking_index is not None and selected_booking_index < len(options):
                    selected_option = options[selected_booking_index]
                    st.markdown(f"### 📅 Book {selected_option['mode']}")
                    st.info(f"Route: {origin} → {route_destination} · Estimated fare: PKR {selected_option['cost']:,} · {trip_type}")
                    with st.form("transport_booking_form"):
                        booking_name = st.text_input("Contact name")
                        booking_phone = st.text_input("Phone number", placeholder="+92 300 1234567")
                        pickup_location = st.text_input("Pickup location", value=origin)
                        travel_date = st.date_input("Travel date")
                        pickup_time = st.time_input("Pickup time")
                        booking_columns = st.columns(2)
                        submit_booking = booking_columns[0].form_submit_button("✅ Confirm booking request", use_container_width=True)
                        cancel_booking = booking_columns[1].form_submit_button("Cancel", use_container_width=True)
                        if cancel_booking:
                            st.session_state.pop("selected_transport_booking", None)
                            st.rerun()
                        if submit_booking:
                            if not booking_name.strip() or not booking_phone.strip() or not pickup_location.strip():
                                st.error("Name, phone number and pickup location are required.")
                            else:
                                reference = save_transport_booking({
                                    "mode": selected_option["mode"], "origin": origin, "destination": route_destination,
                                    "trip_type": trip_type, "travelers": int(travelers), "estimated_cost": selected_option["cost"],
                                    "pickup_location": pickup_location.strip(), "travel_date": travel_date.isoformat(),
                                    "pickup_time": pickup_time.strftime("%H:%M"), "contact_name": booking_name.strip(),
                                    "contact_phone": booking_phone.strip(),
                                })
                                st.session_state.pop("selected_transport_booking", None)
                                st.success(f"Booking request {reference} saved. Status: Pending confirmation.")
                                st.caption("A transport provider must confirm the fare, vehicle and availability before travel.")
            booking_rows = transport_booking_rows()
            if booking_rows:
                st.markdown("#### 📋 My transport booking requests")
                st.dataframe(
                    [{"Reference": f"TRP-{row['id']:06d}", "Vehicle": row["mode"], "Route": f"{row['origin']} → {row['destination']}", "Date": row["travel_date"], "Status": row["status"]} for row in booking_rows[:10]],
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption("Planning estimates only. Confirm live fares, schedules, road conditions and availability before booking.")

        elif hub_view == "📍 Explore Nearby":
            st.markdown(f"### 📍 Explore Nearby: {hub_destination}")
            place_details = nearby_place_details(hub_destination)
            place_names = [place["name"] for place in place_details]
            selected_place = st.session_state.get("selected_nearby_place", place_names[0])
            if selected_place not in place_names:
                selected_place = place_names[0]
            place_columns = st.columns(min(2, len(place_details)))
            for index, place in enumerate(place_details):
                with place_columns[index % len(place_columns)]:
                    if st.button(f"📍 {place['name']}", key=f"nearby_{hub_destination}_{index}", use_container_width=True):
                        st.session_state.selected_nearby_place = place["name"]
                        st.rerun()
            selected = next(place for place in place_details if place["name"] == selected_place)
            st.markdown(f"### 📷 {selected['name']}")
            detail_left, detail_right = st.columns([1.2, 1])
            with detail_left:
                st.image(selected["image"], use_container_width=True)
            with detail_right:
                st.markdown(f"**{selected['description']}**")
                st.markdown(f"📌 **Plan:** {selected['visit']}")
                st.markdown(f"💡 **Visitor tip:** {selected['tip']}")
                st.caption("Photo is loaded from a public image service. Confirm current access, timings and local conditions before visiting.")

        elif hub_view == "📄 Travel Guide":
            st.markdown(f"### 📄 Travel Guide: {hub_destination}")
            guide = travel_guide(hub_destination, hub_entities)
            st.markdown(guide)
            st.download_button("📄 Download Travel Guide", data=guide, file_name=f"{hub_destination.replace(' ', '_')}_guide.md", mime="text/markdown", key="download_guide")

        elif hub_view == "❤️ My Trips":
            st.markdown("### ❤️ My Trips")
            trips = favorite_rows()
            if not trips:
                st.info("No saved trips yet. Generate an itinerary and save it from Trip Tools.")
            for trip in trips:
                with st.expander(f"{trip['destination']} · {trip['duration_days']} days · PKR {trip['total_budget']:,}"):
                    st.markdown(trip["itinerary_markdown"])

        elif hub_view == "📊 Trip Dashboard":
            st.markdown("### 📊 Trip Dashboard")
            trips = favorite_rows()
            if not trips:
                st.info("No saved trips yet. Generate an itinerary and save it from Trip Tools to populate your dashboard.")
            else:
                destination_filter = st.selectbox("Filter by destination", ["All destinations"] + sorted({trip["destination"] for trip in trips}), key="dashboard_destination_filter")
                filtered_trips = trips if destination_filter == "All destinations" else [trip for trip in trips if trip["destination"] == destination_filter]
                total_budget = sum(float(trip["total_budget"]) for trip in filtered_trips)
                average_budget = total_budget / len(filtered_trips)
                average_days = sum(int(trip["duration_days"]) for trip in filtered_trips) / len(filtered_trips)
                destination_counts = {}
                for trip in filtered_trips:
                    destination_counts[trip["destination"]] = destination_counts.get(trip["destination"], 0) + 1
                top_destination = max(destination_counts, key=destination_counts.get)
                dashboard_cols = st.columns(4)
                dashboard_cols[0].metric("Saved trips", len(filtered_trips))
                dashboard_cols[1].metric("Planned budget", f"PKR {total_budget:,.0f}")
                dashboard_cols[2].metric("Average budget", f"PKR {average_budget:,.0f}")
                dashboard_cols[3].metric("Average duration", f"{average_days:.1f} days")
                st.markdown(f"#### Most planned destination: {top_destination}")
                budget_by_destination = {}
                for trip in filtered_trips:
                    budget_by_destination[trip["destination"]] = budget_by_destination.get(trip["destination"], 0) + float(trip["total_budget"])
                st.bar_chart(budget_by_destination, y_label="Budget (PKR)", color="#19a9e8")
                st.markdown("#### Recent trip activity")
                search = st.text_input("Search saved trips", placeholder="Destination name...", key="dashboard_trip_search").strip().casefold()
                visible_trips = [trip for trip in filtered_trips if not search or search in trip["destination"].casefold()]
                table_rows = [{"Destination": trip["destination"], "Days": int(trip["duration_days"]), "Budget (PKR)": float(trip["total_budget"]), "Created": trip["created_at"][:10]} for trip in visible_trips]
                st.dataframe(table_rows, use_container_width=True, hide_index=True)
                export_data = io.StringIO()
                writer = csv.DictWriter(export_data, fieldnames=["Destination", "Days", "Budget (PKR)", "Created"])
                writer.writeheader()
                writer.writerows(table_rows)
                st.download_button("📥 Export dashboard CSV", data=export_data.getvalue(), file_name="trip_dashboard.csv", mime="text/csv", use_container_width=True)

"""

with tab_weather:
    st.subheader("🌤️ Live Weather")
    st.caption("Get live weather information for your selected destination.")
    weather_destination = st.text_input("Enter destination for weather", value=st.session_state.extracted_entities.get("destination", "Lahore"), key="weather_destination")
    if st.button("🌤️ Get Live Weather", type="primary"):
        st.session_state.weather_result = get_weather(weather_destination)
    if st.session_state.weather_result:
        st.markdown(f"<div class='metric'><div class='metric-label'>CURRENT CONDITIONS</div><div class='metric-value'>{st.session_state.weather_result}</div></div>", unsafe_allow_html=True)

with tab_landmark:
    st.subheader("📷 Image & Landmark Recognition")
    st.caption("Upload a nature, mountain, lake or landmark photo. The tool gives the most likely place, visual clues and confidence.")
    upload = st.file_uploader("Upload landmark image", type=["jpg", "jpeg", "png"], key="landmark_upload")
    if upload is not None:
        image_bytes = upload.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash != st.session_state.uploaded_image_hash:
            st.session_state.uploaded_image_hash = image_hash
            st.session_state.landmark_result = None
        st.image(image_bytes, width=360)
        if st.button("📷 Analyze landmark", type="primary"):
            st.session_state.landmark_result = recognize_landmark(image_bytes, get_secret("OPENAI_API_KEY"), get_secret("OPENAI_MODEL"))
    if st.session_state.landmark_result:
        result = st.session_state.landmark_result
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0))))
        st.markdown("#### Analysis confidence")
        st.progress(confidence, text=f"{confidence:.0%} confidence")
        if result["low_confidence"]:
            st.warning(f"Possible match, low confidence: {result['message']}")
            st.markdown(f"**Possible place/category:** {result.get('landmark', result.get('category', 'image'))}  \n**What I see:** {result.get('description', 'The image could not be identified confidently.')}")
        else:
            st.success(f"{result.get('title', result['landmark'])} ({result['confidence']:.0%})")
            st.markdown(f"**Likely place:** {result.get('landmark', result.get('title', 'Image analysis'))}  \n**Category:** {result.get('category', 'image')}  \n**What I see:** {result.get('description', result['message'])}")
        metadata_location = result.get("metadata_location")
        if metadata_location:
            st.info(f"GPS metadata found in the image: {metadata_location['latitude']}, {metadata_location['longitude']}")
            st.map([{"lat": metadata_location["latitude"], "lon": metadata_location["longitude"]}], latitude="lat", longitude="lon", size=40, color="#0e7490", zoom=12)
            st.caption("Location view is embedded inside the app; no external Google Maps navigation is required.")
        else:
            st.caption("No GPS metadata was found. The location above is based on visual evidence only.")
        report = (
            f"Landmark Recognition Report\n\n"
            f"Likely place: {result.get('landmark', 'Unknown')}\n"
            f"Category: {result.get('category', 'Unknown')}\n"
            f"Confidence: {confidence:.0%}\n"
            f"Description: {result.get('description', 'No description available.')}\n"
        )
        if metadata_location:
            report += f"GPS: {metadata_location['latitude']}, {metadata_location['longitude']}\n"
        st.download_button("📄 Download analysis report", data=report, file_name="landmark_analysis.txt", mime="text/plain", key="download_landmark_report")

with tab_status:
    st.subheader("💻 System Status")
    st.caption("Live readiness checks for the local travel planning stack.")
    status_items = [("Streamlit application", True), ("SQLite database", True), ("Local RAG knowledge", bool(load_documents())), ("Weather API key", bool(get_secret("WEATHER_API_KEY"))), ("OpenAI API key", bool(get_secret("OPENAI_API_KEY"))), ("Landmark fallback", True)]
    for label, ready in status_items:
        icon = "✅" if ready else "⚠️"
        st.markdown(f"<div class='metric' style='margin:.5rem 0'><span class='status'>{icon}</span>&nbsp; {label}<span style='float:right;color:#aab5c7'>{'Ready' if ready else 'Needs configuration'}</span></div>", unsafe_allow_html=True)

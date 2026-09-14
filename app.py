"""AI Travel Agent - tabbed Streamlit experience."""
import hashlib
import io
import textwrap
import csv
from math import asin, cos, radians, sin, sqrt
import requests
import streamlit as st
from pathlib import Path

from modules.database import destination_rows, favorite_rows, initialize_database, log_trip, save_favorite, save_transport_booking, transport_booking_rows
from modules.hotel_recommender import DESTINATION_COORDINATES, recommend_hotels
from modules.nlp_processor import classify_intent, extract_entities
from modules.rag_engine import format_context, load_documents, retrieve
from modules.recommender import recommend_destinations, train_lightweight_ranker
from modules.vision import recognize_landmark

st.set_page_config(page_title="AI Travel Agent", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")


def get_secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except Exception:
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
    :root { --navy:#0d172b; --panel:#17243a; --panel2:#1e2d45; --line:#2c3a50; --text:#f5f7fb; --muted:#aab5c7; --blue:#078fd0; --blue2:#19a9e8; --green:#39c985; }
    .stApp { background:var(--navy); color:var(--text); }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#17243a; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding:1.2rem 1.35rem; }
    [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] button svg { color:#19a9e8 !important; fill:#19a9e8 !important; stroke:#19a9e8 !important; opacity:1 !important; }
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

tab_assistant, tab_weather, tab_landmark, tab_status = st.tabs(["🗺️ AI Trip Planner", "🌤️ Live Weather", "📷 Landmark Recognition", "💻 System Status"])

with tab_assistant:
    st.subheader("Plan your next journey")
    st.caption("Generate one complete plan with travel guide, budget, transport, hotels, nearby places, food, packing and daily activities.")
    left, right = st.columns([1.25, .9])
    with left:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        prompt = st.chat_input("e.g. Mujhe Lahore ka 5 din ka trip 100k budget mein chahiye")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            entities = extract_entities(prompt, language)
            entities["budget"] = entities["budget"] or budget_filter
            entities["duration"] = entities["duration"] or duration_filter
            entities["travel_style"] = entities["travel_style"] or style_filter.lower()
            entities["interests"] = sorted(set(entities["interests"] + interests_filter))
            entities["origin"] = entities.get("origin") or origin_filter
            entities["travelers"] = entities.get("travelers") or travelers_filter
            entities["trip_type"] = trip_type_filter
            st.session_state.extracted_entities = entities
            intent = classify_intent(prompt)
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
            st.markdown(f"[Open embedded location in Google Maps]({metadata_location['map_url']})")
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

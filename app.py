"""AI Travel Agent - tabbed Streamlit experience."""
import hashlib
import io
import textwrap
import requests
import streamlit as st

from modules.database import destination_rows, favorite_rows, initialize_database, log_trip, save_favorite
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


def compare_destinations(names: list[str], budget: int, duration: int) -> list[dict]:
    rows = {row["name"]: row for row in destinations}
    result = []
    for name in names:
        row = rows.get(name)
        if row:
            result.append({**row, "trip_estimate": int(float(row["avg_daily_cost"]) * duration), "budget_fit": max(0, min(100, round((1 - abs(float(row["avg_daily_cost"]) * duration - budget) / max(budget, 1)) * 100)))} )
    return result


def transport_options(destination: str) -> list[dict[str, str]]:
    return [
        {"mode": "Private car", "best_for": "Families and flexible stops", "estimate": "PKR 8,000-25,000/day"},
        {"mode": "Bus or coach", "best_for": "Budget travel", "estimate": "Check current operator fares"},
        {"mode": "Domestic flight", "best_for": "Long-distance routes", "estimate": "Check airline fare for your date"},
        {"mode": "Local ride-hailing", "best_for": f"Getting around {destination}", "estimate": "Use live in-app fare before booking"},
    ]


def nearby_places(destination: str) -> list[str]:
    nearby = {"Lahore": ["Badshahi Mosque", "Lahore Fort", "Walled City", "Food Street"], "Islamabad": ["Faisal Mosque", "Daman-e-Koh", "Pakistan Monument", "Saidpur Village"], "Hunza": ["Attabad Lake", "Baltit Fort", "Altit Fort", "Passu Cones"], "Skardu": ["Shangrila Resort", "Upper Kachura Lake", "Deosai Plains", "Mansehra viewpoints"]}
    return nearby.get(destination, [f"Verified local attractions near {destination}", "Local market or cultural center", "A nearby viewpoint", "A recommended day trip"])


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
    .stButton > button { background:transparent; color:var(--text); border:1px solid #40516b; border-radius:7px; font-weight:600; }
    .stButton > button:hover { color:white; border-color:var(--blue2); background:#203653; }
    button[kind="primary"] { background:var(--blue) !important; border-color:var(--blue) !important; }
    [data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stTextInput"] input { background:var(--panel2); border-color:var(--line); color:var(--text); }
    [data-baseweb="input"] input, [data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; caret-color:#19a9e8; }
    [data-baseweb="input"] input::placeholder, [data-testid="stTextInput"] input::placeholder, [data-testid="stChatInput"] textarea::placeholder { color:#b7c2d2 !important; opacity:1 !important; -webkit-text-fill-color:#b7c2d2 !important; }
    [data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea { background:#1e2d45 !important; border-color:#19a9e8 !important; }
    [data-baseweb="select"] span, [data-baseweb="select"] input { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] section, [data-testid="stFileUploader"] small { color:#17304d !important; -webkit-text-fill-color:#17304d !important; }
    [data-testid="stDownloadButton"] button { color:#f5f7fb !important; -webkit-text-fill-color:#f5f7fb !important; background:#078fd0 !important; border-color:#078fd0 !important; }
    [data-testid="stChatMessage"] { background:var(--panel); border:1px solid var(--line); }
    .stAlert { background:var(--panel2); border-color:#40516b; }
    </style>
    """, unsafe_allow_html=True)


def sidebar_controls() -> tuple[str, int, int, str, list[str]]:
    st.sidebar.markdown("<div class='brand'><span class='brand-mark'>⚙️</span><span class='brand-title' style='font-size:1.1rem'>Trip Preferences</span></div>", unsafe_allow_html=True)
    language = st.sidebar.selectbox("Select Language / زبان منتخب کریں", ["English", "Roman Urdu", "Urdu"])
    st.sidebar.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    budget = st.sidebar.slider("Budget (PKR)", min_value=10000, max_value=500000, value=100000, step=5000)
    duration = st.sidebar.number_input("Duration (Days)", min_value=1, max_value=30, value=5)
    style = st.sidebar.selectbox("Travel Style", ["Auto", "Budget Friendly", "Comfort", "Luxury", "Family Friendly", "Adventure"])
    interests = st.sidebar.multiselect("Interests & Activities", ["Adventure", "Mountains", "Nature", "Family", "Food", "Culture", "History", "Photography", "Lakes"], default=[])
    st.sidebar.markdown("<div class='rule'></div><div class='card'><b>💡 Plan with confidence</b><br><span style='color:#aab5c7'>Select your preferences, then ask the AI Travel Assistant for recommendations.</span></div>", unsafe_allow_html=True)
    return language, budget, duration, style, [item.lower() for item in interests]


inject_theme()
initialize_database()
destinations = destination_rows()
train_lightweight_ranker(destinations)
for key, default in {"chat_history": [], "extracted_entities": {}, "recommendations": [], "current_itinerary": "", "uploaded_image_hash": None, "landmark_result": None, "weather_result": ""}.items():
    st.session_state.setdefault(key, default)

language, budget_filter, duration_filter, style_filter, interests_filter = sidebar_controls()
st.markdown("<div class='brand'><span class='brand-mark'>✈️</span><span class='brand-title'>AI Travel Agent</span></div><div class='subtitle'>Your intelligent multilingual travel planning assistant</div>", unsafe_allow_html=True)

tab_assistant, tab_tools, tab_hub, tab_weather, tab_landmark, tab_status = st.tabs(["🗺️ AI Trip Planner", "🧰 Trip Tools", "🧭 Travel Hub", "🌤️ Live Weather", "📷 Landmark Recognition", "💻 System Status"])

with tab_assistant:
    st.subheader("Plan your next journey")
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
            if st.button("🔎 Get Recommendations", type="primary", use_container_width=True):
                if not selected_destination:
                    st.error("Please type a destination first.")
                else:
                    entities.update({"destination": selected_destination, "budget": confirmed_budget, "duration": confirmed_duration})
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
        if st.button("🗺️ Generate grounded itinerary", type="primary"):
            entities = st.session_state.extracted_entities
            context = format_context(retrieve(f"{entities['destination']} {' '.join(entities.get('interests', []))}"))
            itinerary = generate_itinerary(entities, context, get_weather(entities["destination"]))
            st.session_state.current_itinerary = itinerary
            log_trip(entities["destination"], entities["budget"], entities["duration"], itinerary)
            st.rerun()
    if st.session_state.current_itinerary:
        st.markdown(st.session_state.current_itinerary)

with tab_tools:
    st.subheader("🧰 Smart Trip Tools")
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
                itinerary = generate_itinerary(entities, context, get_weather(entities["destination"]))
                st.session_state.current_itinerary = itinerary
                log_trip(entities["destination"], int(entities["budget"]), int(entities["duration"]), itinerary)
                st.rerun()
        favorites = favorite_rows()
        if favorites:
            st.markdown("### ❤️ Saved Favorite Trips")
            for favorite in favorites[:5]:
                st.markdown(f"<div class='metric'><b>{favorite['destination']}</b><br><span style='color:#aab5c7'>{favorite['duration_days']} days · PKR {favorite['total_budget']:,}</span></div>", unsafe_allow_html=True)

    with tab_hub:
        st.subheader("🧭 Travel Hub")
        hub_entities = st.session_state.extracted_entities
        hub_destination = hub_entities.get("destination") or st.text_input("Choose a destination for travel tools", value="Lahore", key="hub_destination")
        hub_view = st.selectbox("Open tool", ["⚖️ Compare Destinations", "🚗 Transport Planner", "📍 Explore Nearby", "📄 Travel Guide", "❤️ My Trips", "📊 Trip Dashboard"], key="hub_view")

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

        elif hub_view == "🚗 Transport Planner":
            st.markdown(f"### 🚗 Transport Planner for {hub_destination}")
            for option in transport_options(hub_destination):
                st.markdown(f"<div class='metric'><b>{option['mode']}</b><br><span>{option['best_for']} · {option['estimate']}</span></div>", unsafe_allow_html=True)
            st.caption("Prices are planning estimates. Confirm current fares with the operator before booking.")

        elif hub_view == "📍 Explore Nearby":
            st.markdown(f"### 📍 Explore Nearby: {hub_destination}")
            for place in nearby_places(hub_destination):
                st.markdown(f"<div class='metric'>📍 {place}</div>", unsafe_allow_html=True)

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
            total_saved = len(trips)
            total_budget = sum(float(trip["total_budget"]) for trip in trips)
            dashboard_cols = st.columns(4)
            dashboard_cols[0].metric("Saved trips", total_saved)
            dashboard_cols[1].metric("Destinations", len({trip["destination"] for trip in trips}))
            dashboard_cols[2].metric("Planned budget", f"PKR {total_budget:,.0f}")
            dashboard_cols[3].metric("Current days", int(hub_entities.get("duration", duration_filter)))
            if trips:
                st.markdown("#### Recent trip activity")
                for trip in trips[:5]:
                    st.markdown(f"- **{trip['destination']}** · {trip['duration_days']} days · PKR {trip['total_budget']:,}")

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
    st.caption("Upload any photo. The tool describes the image and identifies a landmark when one is visible.")
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
        if result["low_confidence"]:
            st.warning(f"Low confidence: {result['message']}")
            st.markdown(f"**Possible category:** {result.get('category', 'image')}  \n**What I see:** {result.get('description', 'The image could not be identified confidently.')}")
        else:
            st.success(f"{result.get('title', result['landmark'])} ({result['confidence']:.0%})")
            st.markdown(f"**Category:** {result.get('category', 'image')}  \n**What I see:** {result.get('description', result['message'])}")

with tab_status:
    st.subheader("💻 System Status")
    st.caption("Live readiness checks for the local travel planning stack.")
    status_items = [("Streamlit application", True), ("SQLite database", True), ("Local RAG knowledge", bool(load_documents())), ("Weather API key", bool(get_secret("WEATHER_API_KEY"))), ("OpenAI API key", bool(get_secret("OPENAI_API_KEY"))), ("Landmark fallback", True)]
    for label, ready in status_items:
        icon = "✅" if ready else "⚠️"
        st.markdown(f"<div class='metric' style='margin:.5rem 0'><span class='status'>{icon}</span>&nbsp; {label}<span style='float:right;color:#aab5c7'>{'Ready' if ready else 'Needs configuration'}</span></div>", unsafe_allow_html=True)

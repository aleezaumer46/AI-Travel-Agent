# Master Build Prompt — AI Travel Agent (Streamlit, FYP)

Use this single prompt with an AI coding assistant (e.g. Claude Code, Cursor) to build the entire project. It consolidates the PRD, Architecture, Rules, Phases, and Design decisions already made for this project — follow it exactly.

---

## Who this is for / What to build

You are building **"AI Travel Agent"** — a multilingual (English, Urdu, Roman Urdu), AI-powered travel planning assistant. This is a **Final Year Project (FYP)**, so every AI component must be a **real, working model — not mocked or hardcoded**.

**Primary user:** individual travelers (in Pakistan and the wider region) planning domestic/regional trips who are often more comfortable describing what they want in Urdu or Roman Urdu than formal English.
**Secondary user:** academic evaluators assessing the FYP — they need to see genuine model inference in every subsystem (NLP, recommender, CV, RAG).

The core promise: a user chats in English, Urdu, or Roman Urdu about where they want to travel, their budget, and interests, and gets back a grounded, budget-fitting, day-by-day itinerary — optionally after uploading a photo of a landmark to identify it.

---

## Architecture (mandatory)

Build this as a **single standalone Streamlit application**. Do NOT create a separate backend (no FastAPI, no Flask, no Docker, no external API server). All AI/ML logic (NLP, recommendation, computer vision, RAG) must run **in-process** inside the Streamlit app, so it can be hosted for free on Streamlit Community Cloud. Streamlit is both the presentation layer and the execution orchestrator — it imports and calls the Python modules directly.

## Hard constraints / Rules

- Total RAM usage must stay **under 1.0 GB** (Streamlit Community Cloud free-tier limit). Use lightweight/quantized models, and load heavy models **lazily** (only when first needed), cached via `@st.cache_resource` (for models/objects) and `@st.cache_data` (for data).
- In-memory ML inference / intent parsing must run in **< 300ms**; a full RAG + LLM response in **< 3.0s**.
- All secrets (LLM API keys, weather API key, DB credentials) must be read from `.streamlit/secrets.toml` via `st.secrets` — never hardcoded, never committed to the repo.
- Persistence is **SQLite only** (`travel_agent.db`) — no PostgreSQL/MySQL/managed DB.
- Vector search is a **local** FAISS or ChromaDB index under `data/vector_db/` — no managed/cloud vector DB.
- Every AI component (NLP, recommender, CV, RAG) must be real and working — nothing mocked or hardcoded.
- Relevant chunks must be retrieved from RAG **before** any factual claim is generated — ground the LLM, never let it answer from parametric memory alone. If RAG finds nothing, say so explicitly.
- Degrade gracefully: skip (don't block) if the Weather API fails; ask a clarifying question on incomplete entities; flag "low confidence" on weak landmark matches.
- Follow the project structure below exactly — no renaming or restructuring.
- Build phases strictly in sequence — confirm each phase works before starting the next.

## Project structure (create exactly this layout)

```
ai-travel-agent/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── nlp_processor.py
│   ├── recommender.py
│   ├── vision.py
│   ├── rag_engine.py
│   └── database.py
├── data/
│   ├── travel_agent.db
│   ├── travel_data.csv
│   └── vector_db/
└── saved_models/
    ├── xgboost_recommender.pkl
    └── landmark_model.pt
```

## Tech stack

- **App framework / orchestrator:** Streamlit
- **NLP:** fine-tuned DistilBERT or spaCy for intent classification + entity extraction
- **Recommendation:** scikit-learn (cosine similarity, TruncatedSVD) + XGBoost/LightGBM for ranking
- **Computer vision:** ViT-B/16, MobileNetV3, or ResNet-50 (fine-tuned for landmark classification)
- **RAG:** sentence-transformers (`all-MiniLM-L6-v2`) for embeddings; FAISS or ChromaDB for local vector search; LangChain or LlamaIndex for orchestration
- **LLM:** external LLM API (key from `st.secrets`) for final itinerary generation, grounded by RAG context
- **External data:** Weather API (key from `st.secrets`; failures degrade gracefully)
- **Database:** SQLite (`travel_agent.db`)
- **Caching:** `@st.cache_resource` (models/objects), `@st.cache_data` (data)
- **Hosting:** Streamlit Community Cloud (free tier)

## Features to implement

**1. Multilingual chatbot** (`app.py` + `modules/nlp_processor.py`)
- Accept and respond to English, Urdu, and Roman Urdu input in one chat interface.
- Standardize Roman Urdu via regex pre-processing + a semantic dictionary map (e.g. "hunza" → Destination: Hunza; "100k" → Budget: 100,000 PKR) before running intent/entity extraction.
- Classify intent and extract entities: **budget, duration, destination, travel style, interests**.

**2. Hybrid recommendation engine** (`modules/recommender.py`)
- Content-based filtering: cosine similarity over normalized feature vectors.
- Collaborative filtering: TruncatedSVD matrix factorization over historical ratings.
- Ranking: XGBoost/LightGBM model scoring `[user_id, item_id, budget_match, interest_score]`.
- Cache trained weights (`saved_models/xgboost_recommender.pkl`) with `@st.cache_resource`.

**3. Computer vision landmark recognition** (`modules/vision.py`)
- `st.file_uploader` for images.
- ViT-B/16 or MobileNetV3 (or ResNet-50) fine-tuned for landmark classification, loaded **only on first upload**, cached after.
- Return the identified landmark + confidence score.

**4. RAG knowledge retrieval** (`modules/rag_engine.py`)
- Local FAISS (or ChromaDB) index over a real travel dataset, embedded with `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieve relevant chunks before any factual claim is generated.
- Orchestrate via LangChain or LlamaIndex.

**5. Itinerary & budget optimization**
- Combine: extracted entities + recommendation results + RAG context + external Weather API data.
- Generate a structured, **day-by-day itinerary in markdown** that fits the user's explicit budget.
- Log the generated trip to the `Trips` table in SQLite.

**6. Local database** (`modules/database.py`, SQLite)
- Tables: `Users(id, email, name, preferences_json, created_at)`, `Destinations(id, name, province_country, tags, avg_daily_cost, rating)`, `Hotels(id, destination_id, name, price_per_night, star_rating)`, `Activities(id, destination_id, name, cost, duration_hours)`, `Trips(id, user_id, destination_id, total_budget, duration_days)`.

## UX / session design

- **Sidebar:** language selector, budget/duration/style/interests filters, image uploader.
- **Main panel (top to bottom):** chat window → entity-confirmation card (shown before recommending, so the user can confirm/correct extracted entities) → recommendation cards → landmark recognition result → final itinerary.
- **`st.session_state` keys:** `chat_history`, `extracted_entities`, `uploaded_image`, `recommendations`, `current_itinerary`.
- **Interaction flow:** user picks a language → types a free-text trip request → text is standardized and entities extracted → entity-confirmation card shown for the user to confirm/edit → on confirmation, hybrid recommender returns ranked recommendation cards → if an image was uploaded, landmark result shown alongside (name + confidence, or "low confidence") → app combines entities + recommendations + RAG context + weather into the final markdown itinerary and logs the trip to SQLite.
- **Edge cases (must be visible in the UI):** incomplete entities → targeted clarifying question; weak landmark match → explicit "low confidence" label; RAG finds nothing → say so plainly instead of hallucinating; Weather API failure → continue without weather data, never block.
- Low-confidence/missing-data states should be visually distinct from confident results.

## Build order (follow these phases in sequence — confirm each phase works before moving to the next)

1. **Setup & UI** — `app.py`, layout, sidebar, `st.session_state` scaffolding.
2. **NLP & entity engine** — `modules/nlp_processor.py`, Roman Urdu handling, entity extraction.
3. **ML recommender** — `modules/recommender.py`, content-based + collaborative filtering + XGBoost ranking.
4. **SQLite & RAG** — `modules/database.py` schema, `modules/rag_engine.py`, FAISS index build.
5. **Computer vision** — `modules/vision.py`, upload handler, lazy-loaded inference.
6. **LLM & itinerary builder** — wire RAG + Weather API + LLM into final markdown itinerary generation.
7. **Deployment** — push to GitHub, configure `.streamlit/secrets.toml`, deploy to Streamlit Community Cloud, verify RAM/latency budgets in production.

## Deliverable for each step

For every module you build, also give me:
- The code file(s).
- A short note on which requirement/phase it satisfies.
- Any new entries needed in `requirements.txt`.

**Start with Phase 1 (Setup & UI) and wait for my confirmation before moving to Phase 2.**

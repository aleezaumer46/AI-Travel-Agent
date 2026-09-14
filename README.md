# AI Travel Agent

AI Travel Agent is a multilingual, budget-aware travel planning assistant built as a standalone Streamlit application. Users can describe a trip in English, Urdu, or Roman Urdu and receive recommendations, grounded itinerary planning, budget guidance, packing suggestions, food ideas, transport options, and travel guides.

The project is designed as a Final Year Project (FYP) and keeps its core logic in-process so it can run locally or on Streamlit Community Cloud without a separate backend.

## Highlights

- Multilingual travel request parsing for English, Urdu transliteration, and Roman Urdu
- Destination, budget, duration, travel style, and interest extraction
- Content-based destination recommendations with budget-fit scoring
- Local SQLite persistence for trips and favorite itineraries
- Local TF-IDF retrieval over the bundled travel knowledge base
- Grounded day-by-day itinerary generation with optional OpenAI integration
- Live weather lookup with graceful failure handling
- General image and landmark recognition using OpenAI Vision or a local CLIP fallback
- Smart budget breakdown and trip optimization
- Smart packing assistant and destination food guide
- Destination comparison, transport planner, nearby places, and downloadable travel guide
- Advanced transport planner with route distance, travellers, one-way/return fares, travel time, budget fit, and sorting
- Nearest and best hotel finder with rent, contact number, ratings, reviews, distance, and amenities
- PDF itinerary export and saved favorite trips
- Dark, tabbed Streamlit interface with system readiness status

## Application Tools

The application is organized into focused tabs:

| Tool | Purpose |
| --- | --- |
| AI Trip Planner | Chat-based multilingual planning and itinerary generation |
| Trip Tools | Budget planner, packing assistant, food guide, PDF export, and favorites |
| Travel Hub | Compare destinations, hotel finder, transport, nearby places, travel guide, My Trips, and dashboard |
| Live Weather | Current weather for a selected destination |
| Image Recognition | Landmark identification and general image description |
| System Status | Checks for database, RAG data, API keys, and fallback models |

## Technology Stack

- **UI and orchestration:** Streamlit
- **NLP:** Python rule-based normalization and entity extraction
- **Recommendations:** scikit-learn TF-IDF and cosine similarity
- **Retrieval:** Local TF-IDF retrieval over `data/travel_data.csv`
- **Vision:** OpenAI Vision with lazy-loaded Hugging Face CLIP fallback
- **Weather:** OpenWeather-compatible HTTP API
- **Persistence:** SQLite
- **PDF export:** fpdf2
- **Deployment:** Streamlit Community Cloud

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   ├── config.toml
│   ├── secrets.example.toml
│   └── secrets.toml              # local only, ignored by Git
├── modules/
│   ├── database.py
│   ├── nlp_processor.py
│   ├── rag_engine.py
│   ├── recommender.py
│   └── vision.py
├── data/
│   ├── travel_data.csv
│   ├── travel_agent.db           # generated locally
│   └── vector_db/
└── documentation/
	└── MASTER_BUILD_PROMPT.md
```

## Run Locally

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## API Configuration

The application can start without API keys. Copy the example file to create a local secrets file:

```powershell
Copy-Item .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Then add your credentials:

```toml
OPENAI_API_KEY = "your-openai-key"
WEATHER_API_KEY = "your-weather-key"
OPENAI_MODEL = "gpt-4o-mini"
```

Never commit `.streamlit/secrets.toml`. It is excluded by `.gitignore`. If a key is exposed, revoke it and create a replacement immediately.

## Typical Workflow

1. Select language, budget, duration, style, and interests in the sidebar.
2. Describe the trip in the AI Trip Planner.
3. Confirm or type any destination.
4. Review Smart Recommendations.
5. Generate the grounded itinerary.
6. Open Trip Tools for budget, packing, food, PDF, and favorites.
7. Use Travel Hub for comparisons, transport, nearby exploration, guides, and dashboard insights.

## Data and Privacy

- Trip records and favorites are stored in local SQLite.
- API keys are loaded from Streamlit secrets and are never required in source code.
- Uploaded images are processed during the current Streamlit session and are not intentionally stored by the application.
- The local fallback model may download a CLIP checkpoint to the Hugging Face cache on first use.

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app from the repository.
3. Select `app.py` as the main file.
4. Add the values from `secrets.example.toml` in the Streamlit Cloud Secrets panel.
5. Deploy and verify the System Status tab.

For cloud deployment, avoid committing `travel_agent.db` or local model artifacts. The application creates runtime data when needed.

## Validation

Useful local checks:

```powershell
python -m compileall -q app.py modules
python -c "from modules.database import initialize_database; initialize_database(); print('Database OK')"
```

## Scope and Limitations

Recommendations and local RAG answers are grounded in the bundled travel dataset. A destination not present in that dataset can still be planned, but the UI clearly marks missing verified local knowledge. Vision results are confidence-scored and should be verified when the model reports low confidence.

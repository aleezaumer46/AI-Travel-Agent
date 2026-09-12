# AI Travel Agent

A standalone Streamlit FYP application for multilingual, budget-aware travel planning in Pakistan. The app extracts entities from English, Urdu transliteration, and Roman Urdu; ranks destinations; retrieves local travel context; optionally calls an LLM and weather API; recognizes uploaded landmarks with a lazy vision hook; and logs trips to SQLite.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The app runs without API keys. Add local values to `.streamlit/secrets.toml` for `OPENAI_API_KEY`, `OPENAI_MODEL`, and `WEATHER_API_KEY`. Do not commit credentials.

## Build phases covered

1. Setup and Streamlit UI
2. Multilingual NLP and entity extraction
3. Hybrid content/budget recommender with persisted ranking artifact
4. SQLite schema and local TF-IDF RAG retrieval
5. Lazy landmark upload inference boundary with visible low-confidence state
6. Grounded itinerary generation, optional OpenAI and weather integrations, and trip logging
7. Deployment-ready configuration for Streamlit Community Cloud

For a production CV milestone, place a fine-tuned landmark checkpoint behind `modules/vision.py` and add its model-specific dependency; the current app intentionally reports low confidence rather than inventing a landmark.

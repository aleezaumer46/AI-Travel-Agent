"""SQLite persistence and seed data."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "travel_agent.db"
DATA_PATH = ROOT / "data" / "travel_data.csv"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with get_connection() as connection:
        connection.executescript("""
        CREATE TABLE IF NOT EXISTS Users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, name TEXT, preferences_json TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS Destinations (id INTEGER PRIMARY KEY, name TEXT UNIQUE, province_country TEXT, tags TEXT, avg_daily_cost REAL, rating REAL, description TEXT);
        CREATE TABLE IF NOT EXISTS Hotels (id INTEGER PRIMARY KEY, destination_id INTEGER, name TEXT, price_per_night REAL, star_rating REAL);
        CREATE TABLE IF NOT EXISTS Activities (id INTEGER PRIMARY KEY, destination_id INTEGER, name TEXT, cost REAL, duration_hours REAL);
        CREATE TABLE IF NOT EXISTS Trips (id INTEGER PRIMARY KEY, user_id INTEGER, destination_id INTEGER, total_budget REAL, duration_days INTEGER, created_at TEXT, itinerary_markdown TEXT);
        CREATE TABLE IF NOT EXISTS Favorites (id INTEGER PRIMARY KEY, destination TEXT, total_budget REAL, duration_days INTEGER, itinerary_markdown TEXT, created_at TEXT);
        CREATE TABLE IF NOT EXISTS TransportBookings (id INTEGER PRIMARY KEY, mode TEXT, origin TEXT, destination TEXT, trip_type TEXT, travelers INTEGER, estimated_cost REAL, pickup_location TEXT, travel_date TEXT, pickup_time TEXT, contact_name TEXT, contact_phone TEXT, status TEXT, created_at TEXT);
        """)
        count = connection.execute("SELECT COUNT(*) FROM Destinations").fetchone()[0]
        if count == 0 and DATA_PATH.exists():
            for row in pd.read_csv(DATA_PATH).to_dict("records"):
                connection.execute("INSERT INTO Destinations(name, province_country, tags, avg_daily_cost, rating, description) VALUES (?, ?, ?, ?, ?, ?)", tuple(row.values()))


def destination_rows() -> list[dict]:
    initialize_database()
    with get_connection() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM Destinations ORDER BY rating DESC")]


def log_trip(destination: str, budget: int, duration: int, itinerary: str, user_id: int | None = None) -> None:
    initialize_database()
    with get_connection() as connection:
        row = connection.execute("SELECT id FROM Destinations WHERE name = ?", (destination,)).fetchone()
        if row is None:
            connection.execute("INSERT INTO Destinations(name, province_country, tags, avg_daily_cost, rating, description) VALUES (?, ?, ?, ?, ?, ?)", (destination, "User-selected destination", "travel", max(1, budget // max(duration, 1)), 0.0, "User-selected destination without a verified local description."))
            row = connection.execute("SELECT id FROM Destinations WHERE name = ?", (destination,)).fetchone()
        connection.execute("INSERT INTO Trips(user_id, destination_id, total_budget, duration_days, created_at, itinerary_markdown) VALUES (?, ?, ?, ?, ?, ?)", (user_id, row[0] if row else None, budget, duration, datetime.now(timezone.utc).isoformat(), itinerary))


def save_user_preferences(name: str, email: str, preferences: dict) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute("INSERT OR REPLACE INTO Users(email, name, preferences_json, created_at) VALUES (?, ?, ?, ?)", (email, name, json.dumps(preferences), datetime.now(timezone.utc).isoformat()))


def save_favorite(destination: str, budget: int, duration: int, itinerary: str) -> None:
    initialize_database()
    with get_connection() as connection:
        connection.execute("INSERT INTO Favorites(destination, total_budget, duration_days, itinerary_markdown, created_at) VALUES (?, ?, ?, ?, ?)", (destination, budget, duration, itinerary, datetime.now(timezone.utc).isoformat()))


def favorite_rows() -> list[dict]:
    initialize_database()
    with get_connection() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM Favorites ORDER BY created_at DESC")]


def save_transport_booking(booking: dict) -> str:
    """Save a local transport booking request and return its reference."""
    initialize_database()
    with get_connection() as connection:
        cursor = connection.execute(
            """INSERT INTO TransportBookings
            (mode, origin, destination, trip_type, travelers, estimated_cost, pickup_location, travel_date, pickup_time, contact_name, contact_phone, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (booking["mode"], booking["origin"], booking["destination"], booking["trip_type"], booking["travelers"], booking["estimated_cost"], booking["pickup_location"], booking["travel_date"], booking["pickup_time"], booking["contact_name"], booking["contact_phone"], "Pending confirmation", datetime.now(timezone.utc).isoformat()),
        )
        return f"TRP-{cursor.lastrowid:06d}"


def transport_booking_rows() -> list[dict]:
    initialize_database()
    with get_connection() as connection:
        return [dict(row) for row in connection.execute("SELECT * FROM TransportBookings ORDER BY created_at DESC")]

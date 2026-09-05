"""Read-only travel tools: live upstreams plus explicit reproducible fixtures."""
from __future__ import annotations

from datetime import date, timedelta

import httpx
from fastapi import FastAPI, HTTPException

# These are planning allowances, not scraped booking quotes or verified opening hours.
CATALOG = [
    {"id": "albert_park", "name": "Albert Park", "lat": -36.8508, "lon": 174.7675,
     "indoor": False, "price_cents": 0, "source": "https://www.aucklandcouncil.govt.nz/parks-recreation/Pages/park-details.aspx?Location=3"},
    {"id": "waterfront", "name": "Auckland waterfront walk", "lat": -36.8413, "lon": 174.7638,
     "indoor": False, "price_cents": 0, "source": "https://en.wikipedia.org/wiki/Auckland_waterfront"},
    {"id": "domain", "name": "Auckland Domain", "lat": -36.8591, "lon": 174.7757,
     "indoor": False, "price_cents": 0, "source": "https://en.wikipedia.org/wiki/Auckland_Domain"},
    {"id": "gallery", "name": "Auckland Art Gallery", "lat": -36.8510, "lon": 174.7667,
     "indoor": True, "price_cents": 3000, "source": "https://www.aucklandartgallery.com/visit"},
    {"id": "museum", "name": "Auckland War Memorial Museum", "lat": -36.8607, "lon": 174.7778,
     "indoor": True, "price_cents": 3500, "source": "https://www.aucklandmuseum.com/visit"},
    {"id": "maritime", "name": "New Zealand Maritime Museum", "lat": -36.8404, "lon": 174.7637,
     "indoor": True, "price_cents": 3000, "source": "https://www.maritimemuseum.co.nz/visit"},
]
ALLOWANCES = {"food_per_adult_day": 4500, "transport_per_adult_day": 1800,
              "accommodation_per_adult_night": 12000}
app = FastAPI(title="Travel tools", description="Live read-only APIs and explicitly labeled mock fixtures")


def weather_data(start_date: date, days: int, mode: str) -> dict:
    end = start_date + timedelta(days=days-1)
    if mode == "mock":
        return {"data_mode": "mock", "source": "fixture:auckland-weather-v1", "days": [
            {"date": (start_date+timedelta(days=i)).isoformat(), "rain_probability": 85 if i == 0 else 10}
            for i in range(days)]}
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        response = client.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": -36.8485, "longitude": 174.7633, "daily": "precipitation_probability_max",
            "timezone": "Pacific/Auckland", "start_date": start_date.isoformat(), "end_date": end.isoformat()})
        response.raise_for_status()
        data = response.json()["daily"]
    return {"data_mode": "live", "source": str(response.url), "days": [
        {"date": day, "rain_probability": probability}
        for day, probability in zip(data["time"], data["precipitation_probability_max"], strict=True)]}


def attraction_data(mode: str) -> dict:
    evidence = []
    if mode == "live":
        with httpx.Client(timeout=15, follow_redirects=False, headers={"User-Agent": "TechnicalAssessment/0.1 (educational read-only demo)"}) as client:
            response = client.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "format": "json",
                "list": "geosearch", "gscoord": "-36.8485|174.7633", "gsradius": 10000, "gslimit": 30})
            response.raise_for_status()
            evidence = response.json()["query"]["geosearch"]
    return {"data_mode": mode, "source": "https://en.wikipedia.org/w/api.php" if mode == "live" else "fixture:auckland-attractions-v1",
            "nearby_place_evidence": evidence, "catalog": CATALOG, "allowances_cents": ALLOWANCES,
            "price_basis": "planning_estimate", "currency": "NZD",
            "planning_window": {"start": "09:00", "end": "17:00", "basis": "assumed; verify venue hours before travel"},
            "notes": "Allowances are explicit estimates, not current ticket or accommodation quotes. Venue URLs support follow-up verification."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/weather")
def weather(start_date: date, days: int = 2, mode: str = "live"):
    if mode not in {"live", "mock"} or not 1 <= days <= 7:
        raise HTTPException(422, "Invalid mode or day count")
    try:
        return weather_data(start_date, days, mode)
    except Exception as exc:
        raise HTTPException(502, f"Weather unavailable: {type(exc).__name__}") from exc


@app.get("/attractions")
def attractions(mode: str = "live"):
    if mode not in {"live", "mock"}:
        raise HTTPException(422, "Invalid mode")
    try:
        return attraction_data(mode)
    except Exception as exc:
        raise HTTPException(502, f"Attractions unavailable: {type(exc).__name__}") from exc

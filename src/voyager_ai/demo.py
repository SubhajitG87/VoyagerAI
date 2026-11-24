"""Offline payload builders for the demo LLM client, shaped like the structured schemas."""

from __future__ import annotations

from typing import Any

_RESEARCH_TOPICS: dict[str, list[tuple[str, str]]] = {
    "flight": [
        ("Routes", "Direct and one-stop options usually exist from major hubs."),
        ("Timing", "Booking several weeks ahead tends to give better fares."),
        ("Airports", "Check which airport is closest to the city centre."),
    ],
    "visa": [
        ("Entry", "Entry rules depend on your passport, so confirm them before booking."),
        ("Passport", "Passports are usually expected to stay valid for the whole trip."),
        ("Official sources", "Rely on the destination's official immigration site."),
    ],
    "hotel": [
        ("Areas", "Central districts save transit time but cost more per night."),
        ("Booking", "Free-cancellation rates leave room to change plans."),
        ("Value", "Mid-range stays near public transport are often the best value."),
    ],
    "experience": [
        ("Highlights", "Start with the best-known landmarks and add local favourites."),
        ("Pacing", "Leave an unscheduled block each day to avoid fatigue."),
        ("Food", "Local markets and neighbourhood restaurants show regional cooking."),
    ],
}


def demo_research_payload(agent_id: str, destination: str) -> dict[str, Any]:
    topics = _RESEARCH_TOPICS.get(agent_id, _RESEARCH_TOPICS["experience"])
    return {
        "summary": f"Demo {agent_id} notes for {destination}. These are general, unverified tips.",
        "findings": [
            {"topic": topic, "claim": claim, "basis": "model", "source_url": None}
            for topic, claim in topics
        ],
    }


def _slot(title: str, description: str, low: float, high: float) -> dict[str, Any]:
    return {
        "title": title,
        "description": description,
        "place": None,
        "cost": {"low": low, "high": high},
    }


def demo_itinerary_payload(days: int) -> dict[str, Any]:
    return {
        "days": [
            {
                "number": n,
                "title": f"Day {n}",
                "morning": _slot("Morning walk", "Explore a nearby neighbourhood.", 0, 10),
                "afternoon": _slot(
                    "Local sights", "Visit a landmark and take a long lunch.", 10, 30
                ),
                "evening": _slot("Dinner out", "Try a well-reviewed local restaurant.", 20, 50),
            }
            for n in range(1, days + 1)
        ],
        "tips": [
            "Carry a small amount of local cash for markets.",
            "Download offline maps before you head out.",
        ],
        "budget": {
            "items": [
                {"label": "Flights", "low": 400, "high": 900},
                {"label": "Accommodation", "low": 300, "high": 800},
                {"label": "Food", "low": 150, "high": 350},
                {"label": "Activities", "low": 80, "high": 250},
                {"label": "Local transport", "low": 40, "high": 120},
            ]
        },
    }

"""Opt-in smoke test that runs a real plan. Skipped unless VOYAGER_LIVE_PROVIDER is set.

Run it with, for example:
    VOYAGER_LIVE_PROVIDER=nim NVIDIA_API_KEY=... python -m pytest -m live
"""

from __future__ import annotations

import os

import pytest

from voyager_ai.config import Settings
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.planner import TravelPlanner

PROVIDERS = ("nim", "ollama", "openai-compatible")


@pytest.mark.live
def test_live_plan():
    provider = os.environ.get("VOYAGER_LIVE_PROVIDER", "").strip().lower()
    if not provider:
        pytest.skip("set VOYAGER_LIVE_PROVIDER (nim, ollama or openai-compatible) to run")
    assert provider in PROVIDERS, f"VOYAGER_LIVE_PROVIDER must be one of {PROVIDERS}"

    settings = Settings.from_env()
    settings.provider = provider
    # The model map depends on the provider (nim has its own models).
    settings.use_preset(settings.model_preset)

    preferences = TravelPreferences(departure_city="London", nights=3, vibe="food")
    destination = Destination(name="Lisbon", country="Portugal")
    trip = TravelPlanner(settings=settings).plan_trip(preferences, destination)

    assert trip.errors == {}
    assert trip.itinerary is not None
    assert len(trip.itinerary.days) == 4
    findings = [
        finding
        for result in trip.results.values()
        if result.report
        for finding in result.report.findings
    ]
    assert any(finding.basis == "sourced" for finding in findings)

from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.prompts import RESEARCH_CONTEXT_CHARS, agent_prompt, discovery_prompt


def _itinerary_user(prior):
    prefs = TravelPreferences("Mumbai")
    destination = Destination("Tokyo", "Japan", "Chosen by test.")
    return agent_prompt("itinerary", prefs, destination, "", prior).user


def test_itinerary_prompt_keeps_research_beyond_1100_chars():
    hotel = "A" * 3000 + " LUXURY-TIER-MARKER"
    assert "LUXURY-TIER-MARKER" in _itinerary_user({"hotel": hotel})


def test_itinerary_prompt_caps_oversized_research():
    user = _itinerary_user({"flight": "B" * (RESEARCH_CONTEXT_CHARS + 500)})
    assert "B" * RESEARCH_CONTEXT_CHARS not in user
    assert "B" * (RESEARCH_CONTEXT_CHARS - 3) + "..." in user


def test_itinerary_prompt_placeholder_for_missing_research():
    assert "No visa research was available." in _itinerary_user({})


def test_discovery_prompt_uses_product_name():
    system = discovery_prompt(TravelPreferences("Mumbai")).system
    assert "VoyagerAI's travel discovery expert" in system

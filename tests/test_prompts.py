from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.prompts import (
    ITINERARY_MAX_TOKENS,
    RESEARCH_CONTEXT_CHARS,
    agent_prompt,
    discovery_prompt,
    itinerary_max_tokens,
    itinerary_tokens_for_context,
    research_chars_for_context,
)


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


def test_research_budget_scales_with_context_size():
    assert research_chars_for_context(16384) == 8754
    assert research_chars_for_context(8192) == 2610
    assert research_chars_for_context(4096) == 1100
    assert research_chars_for_context(65536) == RESEARCH_CONTEXT_CHARS


def _itinerary(nights):
    prefs = TravelPreferences("Mumbai", nights=nights)
    return agent_prompt("itinerary", prefs, Destination("Tokyo", "Japan", "Test."), "", {})


def test_short_trip_keeps_the_base_reply_budget():
    assert itinerary_max_tokens(8) == ITINERARY_MAX_TOKENS
    assert _itinerary(7).max_tokens == ITINERARY_MAX_TOKENS


def test_long_trip_gets_room_for_every_day():
    # 30 nights is 31 days: 31 * 260 + 1200.
    assert _itinerary(30).max_tokens == 9260


def test_long_trip_asks_for_short_days():
    assert "This is a long trip" in _itinerary(30).user
    assert "This is a long trip" not in _itinerary(7).user


def test_itinerary_budget_is_capped_by_a_small_context():
    # 8192 - 512 instructions - 1466 minimum research.
    assert itinerary_tokens_for_context(31, 8192) == 6214
    assert itinerary_tokens_for_context(31, 16384) == 9260
    assert itinerary_tokens_for_context(31, 4096) == ITINERARY_MAX_TOKENS


def test_research_budget_shrinks_when_the_reply_grows():
    assert research_chars_for_context(16384, 9260) == 4959

from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.prompts import (
    ITINERARY_MAX_TOKENS,
    RESEARCH_CONTEXT_CHARS,
    discovery_prompt,
    itinerary_max_tokens,
    itinerary_prompt,
    itinerary_tokens_for_context,
    render_report_for_prompt,
    research_chars_for_context,
    research_prompt,
)
from voyager_ai.structured import Finding, ResearchReport

DEST = Destination("Tokyo", "Japan", "Chosen by test.")


def test_discovery_prompt_uses_product_name():
    system = discovery_prompt(TravelPreferences("Mumbai")).system
    assert "VoyagerAI's travel discovery expert" in system


def test_research_budget_scales_with_context_size():
    assert research_chars_for_context(16384) == 8754
    assert research_chars_for_context(8192) == 2610
    assert research_chars_for_context(4096) == 1100
    assert research_chars_for_context(65536) == RESEARCH_CONTEXT_CHARS


def test_short_trip_keeps_the_base_reply_budget():
    assert itinerary_max_tokens(8) == ITINERARY_MAX_TOKENS


def test_long_trip_gets_room_for_every_day():
    # 30 nights is 31 days: 31 * 320 + 1200.
    assert itinerary_max_tokens(31) == 11120


def test_itinerary_budget_is_capped_by_a_small_context():
    # 8192 - 512 instructions - 1466 minimum research.
    assert itinerary_tokens_for_context(31, 8192) == 6214
    assert itinerary_tokens_for_context(31, 16384) == 11120
    assert itinerary_tokens_for_context(31, 4096) == ITINERARY_MAX_TOKENS


def test_research_budget_shrinks_when_the_reply_grows():
    assert research_chars_for_context(16384, 11120) == 3564


def test_experience_prompt_uses_the_travel_style():
    prefs = TravelPreferences("Mumbai", vibe="food")
    assert prefs.vibe_description in research_prompt("experience", prefs, DEST, "").system


def test_visa_prompt_names_nationality_and_official_sources():
    p = research_prompt("visa", TravelPreferences("Mumbai", nationality="Indian"), DEST, "")
    assert "Indian" in p.user and "Prioritise official government sources." in p.system


def test_research_prompts_ask_for_json_and_keep_sampling():
    prefs = TravelPreferences("Mumbai")
    expected = {
        "flight": (0.2, 2400),
        "visa": (0.1, 2600),
        "hotel": (0.25, 2600),
        "experience": (0.35, 2800),
    }
    for agent_id, (temperature, max_tokens) in expected.items():
        p = research_prompt(agent_id, prefs, DEST, "CTX")
        assert p.system.endswith(
            "Return only JSON that matches the schema. Put each fact in its own finding."
        )
        assert "CTX" in p.user
        assert (p.temperature, p.max_tokens) == (temperature, max_tokens)


def test_report_rendering_marks_basis():
    report = ResearchReport(
        "S",
        [
            Finding("Entry", "Visa-free.", "sourced", "https://g.gov"),
            Finding("Fees", "None.", "model"),
        ],
    )
    text = render_report_for_prompt(report, "missing")
    assert text.startswith("S\n")
    assert "- [sourced] Visa-free. (https://g.gov)" in text and "- [model] None." in text


def test_report_rendering_returns_placeholder_when_missing():
    assert render_report_for_prompt(None, "missing") == "missing"


def test_itinerary_prompt_placeholders_truncation_and_budget_labels():
    prompt = itinerary_prompt(
        TravelPreferences("Mumbai", nights=7),
        DEST,
        {"flight": ResearchReport("F" * 5000, [])},
        research_chars=1200,
        max_tokens=4200,
    )
    assert "No visa research was available." in prompt.user
    assert "F" * 1197 + "..." in prompt.user and "F" * 1201 not in prompt.user
    assert "Mid-range hotel (7 nights)" in prompt.user and prompt.max_tokens == 4200
    for label in ("Return economy flights", "Food", "Activities", "Local transport"):
        assert label in prompt.user
    assert "This is a long trip" not in prompt.user


def test_itinerary_prompt_long_trip_asks_for_short_days():
    prompt = itinerary_prompt(TravelPreferences("Mumbai", nights=30), DEST, {}, 1200, 4200)
    assert "This is a long trip" in prompt.user

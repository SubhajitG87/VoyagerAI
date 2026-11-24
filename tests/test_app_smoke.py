from pathlib import Path

from streamlit.testing.v1 import AppTest

from voyager_ai.app import MAX_AVOID_DESTINATIONS, destination_card_html, merge_avoid_list
from voyager_ai.models import Destination, VoyagerError
from voyager_ai.planner import TravelPlanner
from voyager_ai.render import ESTIMATE_NOTE, UNSOURCED_LABEL, VISA_NOTE
from voyager_ai.structured import BudgetItem, CostRange, Finding

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "voyager_ai" / "app.py"


def test_sidebar_shows_env_override(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    monkeypatch.setenv("VOYAGER_AGENT_MODEL_VISA", "custom-visa")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert "visa: custom-visa" in [block.value for block in app.sidebar.code]


def test_destination_card_escapes_model_text():
    card = destination_card_html(
        Destination(name="<b>X</b>", country="Y", why="z", tags=["<i>"], flag="<script>")
    )
    assert "<script>" not in card
    assert "<b>X</b>" not in card
    assert "&lt;script&gt;" in card


def test_avoid_list_drops_duplicates_and_keeps_recent_names():
    assert merge_avoid_list(["Tokyo", "Lisbon"], ["tokyo", "Seoul"]) == ["Lisbon", "tokyo", "Seoul"]
    many = [f"City {i}" for i in range(40)]
    merged = merge_avoid_list([], many)
    assert len(merged) == MAX_AVOID_DESTINATIONS
    assert merged[-1] == "City 39"


def test_failed_refresh_shows_error_and_keeps_cards(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Mumbai")
    app.button[0].click().run()
    assert len(app.session_state.destinations) == 8

    def fail(self, preferences, avoid=()):
        raise VoyagerError("discovery is down")

    monkeypatch.setattr(TravelPlanner, "discover_destinations", fail)
    refresh = next(b for b in app.button if b.label == "Different destinations")
    refresh.click().run()
    assert not app.exception
    assert "discovery is down" in [e.value for e in app.error]
    assert len(app.session_state.destinations) == 8
    assert app.session_state.avoid_destinations == []


def test_full_demo_flow_shows_structured_trip(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Mumbai")
    app.button[0].click().run()
    next(b for b in app.button if b.label == "Plan Lisbon").click().run()
    assert not app.exception
    assert any(e.label.startswith("Day 1 - ") for e in app.expander)
    assert any(UNSOURCED_LABEL in m.value for m in app.markdown)
    assert any(VISA_NOTE in i.value for i in app.info)
    assert [t.label for t in app.tabs] == ["Itinerary", "Research", "Budget", "Markdown"]
    assert len(app.table) == 1
    assert any("# VoyagerAI Trip Plan:" in block.value for block in app.code)
    assert sum(ESTIMATE_NOTE in c.value for c in app.caption) == 2


def test_missing_itinerary_shows_error_instead_of_crashing(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Mumbai")
    app.button[0].click().run()
    next(b for b in app.button if b.label == "Plan Lisbon").click().run()
    trip = app.session_state.trip
    trip.results["itinerary"].itinerary = None
    trip.results["itinerary"].error = "model returned nothing"
    app.session_state.trip = trip
    app.run()
    assert not app.exception
    assert "model returned nothing" in [e.value for e in app.error]
    assert any("No budget" in i.value for i in app.info)


def _planned_app(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    app.text_input[0].input("Mumbai")
    app.button[0].click().run()
    next(b for b in app.button if b.label == "Plan Lisbon").click().run()
    return app


def test_model_text_is_escaped_in_the_app(monkeypatch):
    app = _planned_app(monkeypatch)
    trip = app.session_state.trip
    trip.results["visa"].report.findings.append(
        Finding("Entry", "See [official site](https://evil.example).", "model")
    )
    trip.results["visa"].report.summary = "Costs $50 to $80."
    trip.itinerary.budget.items.append(BudgetItem("[x](https://evil.example)", CostRange(1, 2)))
    app.session_state.trip = trip
    app.run()
    assert not app.exception
    shown = [m.value for m in app.markdown]
    assert not any("](https://evil.example)" in value for value in shown)
    assert "Costs \\$50 to \\$80." in shown
    table = app.table[0].value
    assert (
        not table.astype(str)
        .apply(lambda col: col.str.contains("](https://", regex=False))
        .any()
        .any()
    )


def test_planner_build_error_is_shown_not_raised(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "nim")
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any("NVIDIA_API_KEY" in e.value for e in app.error)
    assert not app.text_input


def test_bad_schema_mode_is_shown_not_raised(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("VOYAGER_SCHEMA_MODE", "nope")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert any("VOYAGER_SCHEMA_MODE" in e.value for e in app.error)

from pathlib import Path

from streamlit.testing.v1 import AppTest

from voyager_ai.app import MAX_AVOID_DESTINATIONS, destination_card_html, merge_avoid_list
from voyager_ai.models import Destination, VoyagerError
from voyager_ai.planner import TravelPlanner

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

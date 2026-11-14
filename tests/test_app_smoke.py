from pathlib import Path

from streamlit.testing.v1 import AppTest

from voyager_ai.app import destination_card_html
from voyager_ai.models import Destination

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

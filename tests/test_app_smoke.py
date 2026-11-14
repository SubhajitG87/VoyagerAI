from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "voyager_ai" / "app.py"


def test_sidebar_shows_env_override(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "demo")
    monkeypatch.setenv("VOYAGER_ENABLE_SEARCH", "false")
    monkeypatch.setenv("VOYAGER_AGENT_MODEL_VISA", "custom-visa")
    app = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    assert not app.exception
    assert "visa: custom-visa" in [block.value for block in app.sidebar.code]

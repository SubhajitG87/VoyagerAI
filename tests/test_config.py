from voyager_ai.config import MODEL_PRESETS, Settings, resolve_model_map


def test_override_survives_preset_change():
    env = {"VOYAGER_AGENT_MODEL_VISA": "custom-visa"}
    model_map = resolve_model_map("small", env)
    assert model_map["visa"] == "custom-visa"
    assert model_map["flight"] == "phi4-mini"


def test_blank_override_is_ignored():
    assert resolve_model_map("small", {"VOYAGER_AGENT_MODEL_VISA": "  "})["visa"] == "phi4-mini"


def test_unknown_preset_falls_back_to_balanced():
    assert resolve_model_map("nope", {}) == MODEL_PRESETS["balanced"]


def test_use_preset_keeps_env_override(monkeypatch):
    monkeypatch.setenv("VOYAGER_AGENT_MODEL_VISA", "custom-visa")
    settings = Settings()
    settings.use_preset("quality")
    assert settings.model_preset == "quality"
    assert settings.model_map["visa"] == "custom-visa"
    assert settings.model_map["itinerary"] == "llama3.3:70b"


def test_ollama_num_ctx_from_env(monkeypatch):
    monkeypatch.setenv("VOYAGER_OLLAMA_NUM_CTX", "8192")
    assert Settings.from_env().ollama_num_ctx == 8192


def test_ollama_num_ctx_bad_value_uses_default(monkeypatch):
    monkeypatch.setenv("VOYAGER_OLLAMA_NUM_CTX", "lots")
    assert Settings.from_env().ollama_num_ctx == 16384

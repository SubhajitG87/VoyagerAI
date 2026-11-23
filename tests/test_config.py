from voyager_ai.config import (
    AGENT_IDS,
    MODEL_PRESETS,
    NIM_BASE_URL,
    NIM_MODELS,
    Settings,
    resolve_model_map,
)


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


def test_ollama_num_ctx_has_a_4096_floor(monkeypatch):
    monkeypatch.setenv("VOYAGER_OLLAMA_NUM_CTX", "1024")
    assert Settings.from_env().ollama_num_ctx == 4096


def test_research_context_chars_from_env(monkeypatch):
    assert Settings.from_env().research_context_chars == 0
    monkeypatch.setenv("VOYAGER_RESEARCH_CONTEXT_CHARS", "5000")
    assert Settings.from_env().research_context_chars == 5000


def test_retry_and_think_settings(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_RETRIES", "0")
    monkeypatch.setenv("VOYAGER_OLLAMA_THINK", "maybe")
    s = Settings.from_env()
    assert s.llm_retries == 1 and s.ollama_think == "false"


def test_nim_model_map_keeps_overrides():
    env = {"VOYAGER_AGENT_MODEL_VISA": "custom/visa"}
    model_map = resolve_model_map("small", env, provider="nim")
    assert model_map["visa"] == "custom/visa"
    assert model_map["flight"] == NIM_MODELS["flight"]


def test_nim_models_cover_every_agent():
    assert set(NIM_MODELS) == set(AGENT_IDS)


def test_nim_settings_from_env(monkeypatch):
    monkeypatch.setenv("VOYAGER_LLM_PROVIDER", "nim")
    for name in AGENT_IDS:
        monkeypatch.delenv(f"VOYAGER_AGENT_MODEL_{name.upper()}", raising=False)
        monkeypatch.delenv(f"VOYAGER_MODEL_{name.upper()}", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.setenv("VOYAGER_SCHEMA_MODE", "vllm_guided")
    settings = Settings.from_env()
    assert settings.nvidia_api_key == "k"
    assert settings.nvidia_base_url == NIM_BASE_URL
    assert settings.schema_mode == "vllm_guided"
    assert settings.model_map == NIM_MODELS


def test_use_preset_keeps_nim_models():
    settings = Settings(provider="nim")
    settings.use_preset("small")
    assert settings.model_map == NIM_MODELS


def test_fetch_settings_defaults_and_env(monkeypatch):
    settings = Settings.from_env()
    assert (settings.fetch_pages, settings.page_chars) == (3, 4000)
    monkeypatch.setenv("VOYAGER_FETCH_PAGES", "0")
    monkeypatch.setenv("VOYAGER_PAGE_CHARS", "1200")
    settings = Settings.from_env()
    assert (settings.fetch_pages, settings.page_chars) == (0, 1200)


def test_fetch_settings_have_floors(monkeypatch):
    monkeypatch.setenv("VOYAGER_FETCH_PAGES", "-2")
    monkeypatch.setenv("VOYAGER_PAGE_CHARS", "10")
    settings = Settings.from_env()
    assert (settings.fetch_pages, settings.page_chars) == (0, 500)

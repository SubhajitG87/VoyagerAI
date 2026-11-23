"""Runtime configuration for VoyagerAI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

MODEL_PRESETS = {
    "small": {
        "discovery": "qwen3:8b",
        "flight": "phi4-mini",
        "visa": "phi4-mini",
        "hotel": "qwen3:8b",
        "experience": "qwen3:8b",
        "itinerary": "qwen3:8b",
    },
    "balanced": {
        "discovery": "qwen3:14b",
        "flight": "mistral-small3.1:24b",
        "visa": "mistral-small3.1:24b",
        "hotel": "qwen3:14b",
        "experience": "qwen3:14b",
        "itinerary": "mistral-small3.1:24b",
    },
    "quality": {
        "discovery": "qwen3:30b",
        "flight": "mistral-small3.1:24b",
        "visa": "llama3.3:70b",
        "hotel": "qwen3:30b",
        "experience": "qwen3:30b",
        "itinerary": "llama3.3:70b",
    },
}

MODEL_RATIONALE = {
    "qwen3": "Strong structured output, reasoning, and multilingual travel ideation.",
    "mistral-small3.1": "Good long-context synthesis for sourced research summaries.",
    "llama3.3": "High-quality larger-model reasoning when local hardware can support it.",
    "phi4-mini": "Small, quick fallback for laptops or constrained environments.",
}

AGENT_IDS = ("discovery", "flight", "visa", "hotel", "experience", "itinerary")


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def resolve_model_map(preset: str, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return a preset's agent models with per-agent environment overrides applied."""

    source = os.environ if env is None else env
    model_map = dict(MODEL_PRESETS.get(preset, MODEL_PRESETS["balanced"]))
    for agent_id in AGENT_IDS:
        for prefix in ("VOYAGER_AGENT_MODEL_", "VOYAGER_MODEL_"):
            value = (source.get(f"{prefix}{agent_id.upper()}") or "").strip()
            if value:
                model_map[agent_id] = value
                break
    return model_map


@dataclass
class Settings:
    """Configurable runtime settings.

    The defaults are intentionally local-first: Ollama for LLM inference and
    DuckDuckGo snippets for lightweight public web context.
    """

    provider: str = "ollama"
    model_preset: str = "balanced"
    model_map: dict[str, str] = field(default_factory=lambda: dict(MODEL_PRESETS["balanced"]))
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 16384
    # Characters of research per agent passed to the itinerary agent. 0 means automatic.
    research_context_chars: int = 0
    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = ""
    request_timeout_s: int = 180
    search_provider: str = "duckduckgo"
    search_enabled: bool = True
    search_max_results: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv_if_available()
        preset = os.getenv("VOYAGER_MODEL_PRESET", "balanced").strip().lower()
        if preset not in MODEL_PRESETS:
            preset = "balanced"

        return cls(
            provider=os.getenv("VOYAGER_LLM_PROVIDER", "ollama").strip().lower(),
            model_preset=preset,
            model_map=resolve_model_map(preset),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_num_ctx=max(4096, _env_int("VOYAGER_OLLAMA_NUM_CTX", 16384)),
            research_context_chars=max(0, _env_int("VOYAGER_RESEARCH_CONTEXT_CHARS", 0)),
            openai_base_url=os.getenv(
                "OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"
            ).rstrip("/"),
            openai_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            request_timeout_s=_env_int("VOYAGER_REQUEST_TIMEOUT_S", 180),
            search_provider=os.getenv("VOYAGER_SEARCH_PROVIDER", "duckduckgo").strip().lower(),
            search_enabled=_env_bool("VOYAGER_ENABLE_SEARCH", True),
            search_max_results=max(1, _env_int("VOYAGER_SEARCH_MAX_RESULTS", 4)),
        )

    def use_preset(self, preset: str) -> None:
        """Switch preset while keeping per-agent environment overrides."""

        self.model_preset = preset if preset in MODEL_PRESETS else "balanced"
        self.model_map = resolve_model_map(self.model_preset)

    def model_for(self, agent_id: str) -> str:
        return self.model_map.get(agent_id, self.model_map.get("itinerary", "qwen3:8b"))


def load_settings() -> Settings:
    return Settings.from_env()

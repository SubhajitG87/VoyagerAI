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

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# UNPROBED placeholders (2026-11-23): run scripts/probe_schema.py with NVIDIA_API_KEY and apply the selection rule in docs/superpowers/plans/2026-11-23-voyagerai-phase-1-engine.md, Task 3 Step 6.  # noqa: E501
NIM_MODELS = {
    "discovery": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "flight": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "visa": "mistralai/mistral-large-4-1t-a52b",
    "hotel": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "experience": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "itinerary": "mistralai/mistral-large-4-1t-a52b",
}
NIM_SCHEMA_MODE = "json_schema"

# Model-call attempts. NIM's free endpoint allows about 40 requests a minute account-wide,
# so it gets more attempts to ride out a quota window.
DEFAULT_LLM_RETRIES = 3
NIM_LLM_RETRIES = 5

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


def _env_think() -> str:
    value = os.getenv("VOYAGER_OLLAMA_THINK", "false").strip().lower()
    return value if value in {"false", "true", "default"} else "false"


def llm_retries_for(provider: str) -> int:
    """Attempts per model call: VOYAGER_LLM_RETRIES if set, else the provider's default."""

    default = NIM_LLM_RETRIES if provider.strip().lower() == "nim" else DEFAULT_LLM_RETRIES
    return max(1, _env_int("VOYAGER_LLM_RETRIES", default))


def resolve_model_map(
    preset: str, env: Mapping[str, str] | None = None, provider: str = "ollama"
) -> dict[str, str]:
    """Return a preset's agent models with per-agent environment overrides applied.

    The nim provider has one fixed model per agent whatever the preset.
    """

    source = os.environ if env is None else env
    if provider == "nim":
        model_map = dict(NIM_MODELS)
    else:
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
    nvidia_api_key: str = ""
    nvidia_base_url: str = NIM_BASE_URL
    # Structured-output mode. "" means the provider default.
    schema_mode: str = ""
    request_timeout_s: int = 180
    # Attempts per model call on 429, 5xx and connection errors. from_env uses llm_retries_for.
    llm_retries: int = DEFAULT_LLM_RETRIES
    # Ollama "think" flag: "false" (default), "true", or "default" to leave it to the model.
    ollama_think: str = "false"
    search_provider: str = "duckduckgo"
    search_enabled: bool = True
    search_max_results: int = 4
    search_retries: int = 2
    # Top search results each research agent reads in full. 0 turns page fetching off.
    fetch_pages: int = 3
    # Characters of readable text kept per fetched page.
    page_chars: int = 4000
    # LLM calls allowed at once. 0 means automatic: 1 on Ollama, 4 elsewhere.
    llm_concurrency: int = 0

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv_if_available()
        preset = os.getenv("VOYAGER_MODEL_PRESET", "balanced").strip().lower()
        if preset not in MODEL_PRESETS:
            preset = "balanced"

        provider = os.getenv("VOYAGER_LLM_PROVIDER", "ollama").strip().lower()
        return cls(
            provider=provider,
            model_preset=preset,
            model_map=resolve_model_map(preset, provider=provider),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_num_ctx=max(4096, _env_int("VOYAGER_OLLAMA_NUM_CTX", 16384)),
            research_context_chars=max(0, _env_int("VOYAGER_RESEARCH_CONTEXT_CHARS", 0)),
            openai_base_url=os.getenv(
                "OPENAI_COMPATIBLE_BASE_URL", "http://localhost:1234/v1"
            ).rstrip("/"),
            openai_api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
            nvidia_api_key=os.getenv("NVIDIA_API_KEY", ""),
            nvidia_base_url=os.getenv("NVIDIA_BASE_URL", NIM_BASE_URL).rstrip("/"),
            schema_mode=os.getenv("VOYAGER_SCHEMA_MODE", "").strip().lower(),
            request_timeout_s=_env_int("VOYAGER_REQUEST_TIMEOUT_S", 180),
            llm_retries=llm_retries_for(provider),
            ollama_think=_env_think(),
            search_provider=os.getenv("VOYAGER_SEARCH_PROVIDER", "duckduckgo").strip().lower(),
            search_enabled=_env_bool("VOYAGER_ENABLE_SEARCH", True),
            search_max_results=max(1, _env_int("VOYAGER_SEARCH_MAX_RESULTS", 4)),
            search_retries=max(1, _env_int("VOYAGER_SEARCH_RETRIES", 2)),
            fetch_pages=max(0, _env_int("VOYAGER_FETCH_PAGES", 3)),
            page_chars=max(500, _env_int("VOYAGER_PAGE_CHARS", 4000)),
            llm_concurrency=max(0, _env_int("VOYAGER_LLM_CONCURRENCY", 0)),
        )

    def use_preset(self, preset: str) -> None:
        """Switch preset while keeping per-agent environment overrides."""

        self.model_preset = preset if preset in MODEL_PRESETS else "balanced"
        self.model_map = resolve_model_map(self.model_preset, provider=self.provider)

    @property
    def effective_llm_concurrency(self) -> int:
        """LLM calls allowed at once; a local Ollama server handles one at a time."""

        if self.llm_concurrency > 0:
            return self.llm_concurrency
        return 1 if self.provider == "ollama" else 4

    def model_for(self, agent_id: str) -> str:
        return self.model_map.get(agent_id, self.model_map.get("itinerary", "qwen3:8b"))


def load_settings() -> Settings:
    return Settings.from_env()

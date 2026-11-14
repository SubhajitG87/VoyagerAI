"""LLM clients for local and open-model backends."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from voyager_ai.config import Settings
from voyager_ai.models import VoyagerError


class LLMError(VoyagerError):
    """Raised when an LLM provider cannot complete a request."""


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any]


class BaseLLMClient:
    def complete(
        self,
        *,
        agent_id: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.25,
        max_tokens: int = 2800,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    """HTTP client for Ollama's local chat API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def complete(
        self,
        *,
        agent_id: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.25,
        max_tokens: int = 2800,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            import httpx
        except Exception as exc:
            raise LLMError("Install httpx to use the Ollama provider: pip install httpx") from exc

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception as exc:
            hint = (
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` and "
                f"pull the model with `ollama pull {model}`."
            )
            raise LLMError(f"{hint} Provider error: {exc}") from exc

        data = response.json()
        text = (data.get("message") or {}).get("content") or data.get("response") or ""
        if not text.strip():
            raise LLMError(f"Ollama returned an empty response for {agent_id} using {model}.")
        return LLMResponse(text=text.strip(), model=model, raw=data)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for local OpenAI-compatible servers such as LM Studio, vLLM, or llama.cpp."""

    def __init__(self, base_url: str, api_key: str = "", timeout_s: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def complete(
        self,
        *,
        agent_id: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.25,
        max_tokens: int = 2800,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        try:
            import httpx
        except Exception as exc:
            raise LLMError(
                "Install httpx to use OpenAI-compatible providers: pip install httpx"
            ) from exc

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception as exc:
            raise LLMError(
                f"OpenAI-compatible provider failed at {self.base_url} with model {model}: {exc}"
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        if not text.strip():
            raise LLMError(
                f"OpenAI-compatible provider returned an empty response for {agent_id} "
                f"using {model}."
            )
        return LLMResponse(text=text.strip(), model=model, raw=data)


class DemoLLMClient(BaseLLMClient):
    """Deterministic offline responses for smoke tests and UI demos."""

    def complete(
        self,
        *,
        agent_id: str,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.25,
        max_tokens: int = 2800,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if agent_id == "discovery":
            text = json.dumps({"destinations": _demo_destinations()})
        elif agent_id == "itinerary":
            text = _demo_itinerary(user)
        else:
            text = _demo_agent(agent_id, user)
        return LLMResponse(text=text, model="demo", raw={"provider": "demo"})


def build_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.provider.lower()
    if provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.request_timeout_s)
    if provider in {"openai-compatible", "openai_compatible", "openai"}:
        return OpenAICompatibleClient(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.request_timeout_s,
        )
    if provider == "demo":
        return DemoLLMClient()
    raise LLMError(f"Unsupported VOYAGER_LLM_PROVIDER: {settings.provider}")


def _demo_destinations() -> list[dict[str, Any]]:
    return [
        {
            "name": "Lisbon",
            "country": "Portugal",
            "flag": "PT",
            "why": (
                "Sunny viewpoints, walkable neighbourhoods, and Atlantic day trips make it an "
                "easy first plan."
            ),
            "tags": ["coastal city", "food", "culture"],
        },
        {
            "name": "Tokyo",
            "country": "Japan",
            "flag": "JP",
            "why": (
                "Layered neighbourhoods, seasonal food, and smooth transit reward both first-"
                "timers and repeat visitors."
            ),
            "tags": ["city", "food", "design"],
        },
        {
            "name": "Cape Town",
            "country": "South Africa",
            "flag": "ZA",
            "why": (
                "Mountain views, beaches, wine country, and wildlife day trips create a varied "
                "week without rushing."
            ),
            "tags": ["nature", "wine", "coast"],
        },
        {
            "name": "Athens",
            "country": "Greece",
            "flag": "GR",
            "why": (
                "Ancient sites, neighbourhood dining, and island extensions make it compact "
                "but rich."
            ),
            "tags": ["history", "food", "islands"],
        },
        {
            "name": "Seoul",
            "country": "South Korea",
            "flag": "KR",
            "why": (
                "Palaces, design districts, late-night markets, and mountain trails fit a high-"
                "energy city escape."
            ),
            "tags": ["city", "markets", "culture"],
        },
        {
            "name": "Barcelona",
            "country": "Spain",
            "flag": "ES",
            "why": (
                "Architecture, beaches, tapas routes, and easy rail trips keep every day distinct."
            ),
            "tags": ["architecture", "beach", "food"],
        },
        {
            "name": "Singapore",
            "country": "Singapore",
            "flag": "SG",
            "why": (
                "A polished, food-forward hub with gardens, museums, and regional stopover "
                "potential."
            ),
            "tags": ["food", "urban", "easy transit"],
        },
        {
            "name": "Vancouver",
            "country": "Canada",
            "flag": "CA",
            "why": (
                "Waterfront neighbourhoods, forests, and mountain day trips make it ideal for "
                "soft adventure."
            ),
            "tags": ["outdoors", "city", "scenic"],
        },
    ]


def _demo_agent(agent_id: str, user: str) -> str:
    labels = {
        "flight": "Flight Research",
        "visa": "Visa And Entry Notes",
        "hotel": "Hotel Shortlist",
        "experience": "Experience Ideas",
    }
    title = labels.get(agent_id, agent_id.title())
    return (
        f"## {title}\n"
        "- Demo mode is active, so this is a deterministic placeholder rather than live "
        "research.\n"
        "- The production path will add search snippets, ask the configured open model to "
        "synthesize them, and preserve source links in the final markdown.\n"
        f"- Request summary: {user[:500].replace(chr(10), ' ')}\n"
    )


def _demo_itinerary(user: str) -> str:
    days_match = re.search(r"\((\d+)\s+days\)", user)
    days = int(days_match.group(1)) if days_match else 8
    lines = []
    for day in range(1, min(days, 10) + 1):
        lines.extend(
            [
                f"## Day {day} - Demo Day {day}",
                "Morning: Start with a neighbourhood walk and a relaxed breakfast.",
                "Afternoon: Visit one anchor attraction and leave buffer time for transit.",
                "Evening: Choose a local dinner spot close to your hotel.",
                "",
            ]
        )
    lines.extend(
        [
            "## Practical Tips",
            (
                "- Verify visas, passport validity, and health rules on official government "
                "sites before booking."
            ),
            "- Keep one flexible block every two days for weather or energy changes.",
            "",
            "## Budget Breakdown",
            "- Flights: Confirm with live fare search.",
            "- Hotels: Price against your exact dates.",
            "- Activities: Pre-book only the scarce or timed-entry experiences.",
        ]
    )
    return "\n".join(lines)

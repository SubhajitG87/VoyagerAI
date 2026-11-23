"""LLM clients for local and open-model backends."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

from voyager_ai.config import NIM_SCHEMA_MODE, Settings
from voyager_ai.models import VoyagerError
from voyager_ai.parser import strip_reasoning
from voyager_ai.retry import RetryableError, parse_retry_after, retry_call

# HTTP statuses a server returns when it does not understand a json_schema response format.
SCHEMA_REJECTED_STATUSES = {400, 422}

# Transient statuses worth retrying. ReadTimeout is deliberately not retried: a 180 s timeout
# would otherwise triple.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


def _check_response(response: httpx.Response) -> httpx.Response:
    """Turn transient statuses into RetryableError; raise other HTTP errors as usual."""

    if response.status_code in RETRYABLE_STATUSES:
        raise RetryableError(
            f"HTTP {response.status_code}",
            retry_after=parse_retry_after(response.headers.get("Retry-After")),
        )
    response.raise_for_status()
    return response


class LLMError(VoyagerError):
    """Raised when an LLM provider cannot complete a request."""


SchemaMode = Literal["json_schema", "nvext_guided", "vllm_guided", "json_object"]
SCHEMA_MODES = ("json_schema", "nvext_guided", "vllm_guided", "json_object")


def apply_schema(
    payload: dict[str, Any], schema: dict[str, Any], mode: SchemaMode, name: str
) -> dict[str, Any]:
    """Return a copy of ``payload`` that asks the server for output matching ``schema``."""

    body = dict(payload)
    if mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": schema},
        }
    elif mode == "nvext_guided":
        body["nvext"] = {"guided_json": schema}
    elif mode == "vllm_guided":
        body["guided_json"] = schema
    elif mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    else:
        raise LLMError(f"Unknown schema mode: {mode}")
    return body


# Receives each content piece as a streaming client produces it.
ChunkSink = Callable[[str], None]


class _SchemaRejected(Exception):
    """The server refused a json_schema response format before streaming anything."""


# Streaming parsers return (piece, done) for one line; piece is None when the line has no text.
_LineParser = Callable[[str], "tuple[str | None, bool]"]


def _consume_stream(
    provider: str,
    open_stream: Callable[[], Any],
    parse_line: _LineParser,
    on_chunk: ChunkSink,
    rejected_statuses: set[int] | None = None,
) -> str:
    """Run one streaming attempt and return the full text.

    A connection failure before the first chunk is a RetryableError. Any httpx failure after
    the first chunk raises LLMError, because retrying would replay text already shown.
    A status in ``rejected_statuses`` raises _SchemaRejected so the caller can change the request.
    """

    parts: list[str] = []
    emitted = 0
    try:
        with open_stream() as response:
            if rejected_statuses and response.status_code in rejected_statuses:
                raise _SchemaRejected
            _check_response(response)
            for line in response.iter_lines():
                piece, done = parse_line(line)
                if piece:
                    parts.append(piece)
                    emitted += len(piece)
                    on_chunk(piece)
                if done:
                    break
    except httpx.HTTPError as exc:
        if emitted:
            raise LLMError(
                f"{provider} stream interrupted after {emitted} characters: {exc}"
            ) from exc
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)):
            raise RetryableError(str(exc)) from exc
        raise
    return "".join(parts)


def _parse_ollama_line(line: str) -> tuple[str | None, bool]:
    if not line.strip():
        return None, False
    data = json.loads(line)
    return (data.get("message") or {}).get("content") or None, bool(data.get("done"))


def _parse_sse_line(line: str) -> tuple[str | None, bool]:
    if not line.startswith("data: "):
        return None, False
    body = line[len("data: ") :].strip()
    if body == "[DONE]":
        return None, True
    choices = json.loads(body).get("choices") or []
    return ((choices[0].get("delta") or {}).get("content") if choices else None) or None, False


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
        on_chunk: ChunkSink | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class OllamaClient(BaseLLMClient):
    """HTTP client for Ollama's local chat API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_s: int = 180,
        num_ctx: int = 16384,
        think: str = "false",
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.num_ctx = num_ctx
        self.think = think
        self.retries = retries
        self.sleep = sleep

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
        on_chunk: ChunkSink | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": on_chunk is not None,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            },
        }
        if self.think in {"true", "false"}:
            payload["think"] = self.think == "true"
        if json_schema is not None:
            payload["format"] = json_schema

        url = f"{self.base_url}/api/chat"

        def attempt() -> httpx.Response:
            try:
                response = httpx.post(url, json=payload, timeout=self.timeout_s)
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                raise RetryableError(str(exc)) from exc
            return _check_response(response)

        def attempt_stream() -> str:
            return _consume_stream(
                "Ollama",
                lambda: httpx.stream("POST", url, json=payload, timeout=self.timeout_s),
                _parse_ollama_line,
                on_chunk,  # type: ignore[arg-type]
            )

        try:
            if on_chunk is not None:
                text = retry_call(attempt_stream, attempts=self.retries, sleep=self.sleep)
                data: dict[str, Any] = {"streamed": True}
            else:
                response = retry_call(attempt, attempts=self.retries, sleep=self.sleep)
                data = response.json()
                text = (data.get("message") or {}).get("content") or data.get("response") or ""
        except LLMError:
            raise
        except Exception as exc:
            hint = (
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` and "
                f"pull the model with `ollama pull {model}`."
            )
            raise LLMError(f"{hint} Provider error: {exc}") from exc

        text = strip_reasoning(text)
        if not text.strip():
            raise LLMError(f"Ollama returned an empty response for {agent_id} using {model}.")
        return LLMResponse(text=text.strip(), model=model, raw=data)


class OpenAICompatibleClient(BaseLLMClient):
    """Client for local OpenAI-compatible servers such as LM Studio, vLLM, or llama.cpp."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: int = 180,
        retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        schema_mode: SchemaMode = "json_schema",
        fallback: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.retries = retries
        self.sleep = sleep
        self.schema_mode = schema_mode
        # Retry with plain JSON mode when the server rejects the schema (HTTP 400 or 422).
        self.fallback = fallback

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
        on_chunk: ChunkSink | None = None,
    ) -> LLMResponse:
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
        if on_chunk is not None:
            payload["stream"] = True
        fallback = payload
        if json_schema is not None:
            name = f"{agent_id}_response"
            payload = apply_schema(payload, json_schema, self.schema_mode, name)
            # Older servers only know JSON mode. The parser still validates the result.
            fallback = apply_schema(fallback, json_schema, "json_object", name)
        can_fall_back = (
            json_schema is not None and self.fallback and self.schema_mode != "json_object"
        )

        url = f"{self.base_url}/chat/completions"

        def attempt() -> httpx.Response:
            try:
                response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if can_fall_back and response.status_code in SCHEMA_REJECTED_STATUSES:
                    response = httpx.post(
                        url, json=fallback, headers=headers, timeout=self.timeout_s
                    )
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                raise RetryableError(str(exc)) from exc
            return _check_response(response)

        def stream_once(body: dict[str, Any], schema_requested: bool) -> str:
            return _consume_stream(
                "OpenAI-compatible provider",
                lambda: httpx.stream(
                    "POST", url, json=body, headers=headers, timeout=self.timeout_s
                ),
                _parse_sse_line,
                on_chunk,  # type: ignore[arg-type]
                SCHEMA_REJECTED_STATUSES if schema_requested else None,
            )

        def attempt_stream() -> str:
            try:
                return stream_once(payload, can_fall_back)
            except _SchemaRejected:
                return stream_once(fallback, False)

        try:
            if on_chunk is not None:
                text = retry_call(attempt_stream, attempts=self.retries, sleep=self.sleep)
                data: dict[str, Any] = {"streamed": True}
            else:
                response = retry_call(attempt, attempts=self.retries, sleep=self.sleep)
                data = response.json()
                choices = data.get("choices") or []
                text = ((choices[0].get("message") or {}).get("content") if choices else "") or ""
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"OpenAI-compatible provider failed at {self.base_url} with model {model}: {exc}"
            ) from exc

        text = strip_reasoning(text)
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
        on_chunk: ChunkSink | None = None,
    ) -> LLMResponse:
        if agent_id == "discovery":
            text = json.dumps({"destinations": _demo_destinations()})
        elif agent_id == "itinerary":
            text = _demo_itinerary(user)
        else:
            text = _demo_agent(agent_id, user)
        if on_chunk is not None:
            size = max(1, -(-len(text) // 3))
            for start in range(0, len(text), size):
                on_chunk(text[start : start + size])
        return LLMResponse(text=text, model="demo", raw={"provider": "demo"})


def _schema_mode(configured: str, default: str) -> SchemaMode:
    mode = configured or default
    if mode not in SCHEMA_MODES:
        raise LLMError(f"Unsupported VOYAGER_SCHEMA_MODE: {mode}. Use one of {SCHEMA_MODES}.")
    return cast(SchemaMode, mode)


def build_llm_client(settings: Settings) -> BaseLLMClient:
    provider = settings.provider.lower()
    if provider == "ollama":
        return OllamaClient(
            settings.ollama_base_url,
            settings.request_timeout_s,
            settings.ollama_num_ctx,
            think=settings.ollama_think,
            retries=settings.llm_retries,
        )
    if provider in {"openai-compatible", "openai_compatible", "openai"}:
        return OpenAICompatibleClient(
            settings.openai_base_url,
            settings.openai_api_key,
            settings.request_timeout_s,
            retries=settings.llm_retries,
            schema_mode=_schema_mode(settings.schema_mode, "json_schema"),
        )
    if provider == "nim":
        if not settings.nvidia_api_key:
            raise LLMError("Set NVIDIA_API_KEY to use the nim provider.")
        return OpenAICompatibleClient(
            settings.nvidia_base_url,
            settings.nvidia_api_key,
            settings.request_timeout_s,
            retries=settings.llm_retries,
            schema_mode=_schema_mode(settings.schema_mode, NIM_SCHEMA_MODE),
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
    for day in range(1, days + 1):
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

"""LLM clients for local and open-model backends."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

from voyager_ai.config import NIM_SCHEMA_MODE, Settings
from voyager_ai.demo import demo_itinerary_payload, demo_research_payload
from voyager_ai.models import VoyagerError
from voyager_ai.parser import strip_reasoning
from voyager_ai.retry import RetryableError, parse_retry_after, retry_call
from voyager_ai.utils import clean_text

logger = logging.getLogger(__name__)

# HTTP statuses a server returns when it does not understand a json_schema response format.
SCHEMA_REJECTED_STATUSES = {400, 422}

# Transient statuses worth retrying. ReadTimeout is deliberately not retried: a 180 s timeout
# would otherwise triple.
RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
# Shortest wait after HTTP 429. NIM's free endpoint counts requests per minute, so a wait of
# a second or two only spends the next attempt.
RATE_LIMIT_MIN_DELAY_S = 15.0
# Characters of a rejected response body quoted in an error.
BODY_PREVIEW_CHARS = 300
# Failures that mean the server could not be reached at all.
_CONNECTION_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError)


def _check_response(response: httpx.Response) -> httpx.Response:
    """Turn transient statuses into RetryableError; raise other HTTP errors as usual."""

    if response.status_code in RETRYABLE_STATUSES:
        raise RetryableError(
            f"HTTP {response.status_code}",
            retry_after=parse_retry_after(response.headers.get("Retry-After")),
            min_delay=RATE_LIMIT_MIN_DELAY_S if response.status_code == 429 else 0.0,
        )
    response.raise_for_status()
    return response


def _status_text(exc: httpx.HTTPStatusError) -> str:
    """'HTTP <status>' plus the start of the response body, which usually says what was wrong."""

    try:
        body = exc.response.text
    except Exception:  # a streamed body that could not be read
        body = ""
    preview = clean_text(body)[:BODY_PREVIEW_CHARS] if isinstance(body, str) else ""
    status = f"HTTP {exc.response.status_code}"
    return f"{status}: {preview}" if preview else status


def _is_connection_failure(exc: BaseException) -> bool:
    return isinstance(exc, _CONNECTION_ERRORS) or isinstance(exc.__cause__, _CONNECTION_ERRORS)


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


# Streaming parsers return (piece, done, truncated) for one line; piece is None when the line
# has no text, and truncated is True when the line says the reply hit the token limit.
_LineParser = Callable[[str], "tuple[str | None, bool, bool]"]


def _consume_stream(
    provider: str,
    open_stream: Callable[[], Any],
    parse_line: _LineParser,
    on_chunk: ChunkSink,
    rejected_statuses: set[int] | None = None,
) -> tuple[str, bool]:
    """Run one streaming attempt and return the full text and whether it was cut off.

    A connection failure before the first chunk is a RetryableError. Any httpx failure after
    the first chunk raises LLMError, because retrying would replay text already shown.
    A status in ``rejected_statuses`` raises _SchemaRejected so the caller can change the request.
    """

    parts: list[str] = []
    emitted = 0
    truncated = False
    try:
        with open_stream() as response:
            if rejected_statuses and response.status_code in rejected_statuses:
                raise _SchemaRejected
            if isinstance(response.status_code, int) and response.status_code >= 400:
                response.read()  # so an error can quote the body
            _check_response(response)
            for line in response.iter_lines():
                piece, done, cut = parse_line(line)
                truncated = truncated or cut
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
    return "".join(parts), truncated


def _parse_ollama_line(line: str) -> tuple[str | None, bool, bool]:
    if not line.strip():
        return None, False, False
    data = json.loads(line)
    piece = (data.get("message") or {}).get("content") or None
    return piece, bool(data.get("done")), data.get("done_reason") == "length"


def _parse_sse_line(line: str) -> tuple[str | None, bool, bool]:
    if not line.startswith("data: "):
        return None, False, False
    body = line[len("data: ") :].strip()
    if body == "[DONE]":
        return None, True, False
    choices = json.loads(body).get("choices") or []
    if not choices:
        return None, False, False
    piece = (choices[0].get("delta") or {}).get("content") or None
    return piece, False, choices[0].get("finish_reason") == "length"


@dataclass
class LLMResponse:
    text: str
    model: str
    raw: dict[str, Any]
    # True when the reply stopped at the token limit rather than finishing.
    truncated: bool = False


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

        def attempt_stream() -> tuple[str, bool]:
            return _consume_stream(
                "Ollama",
                lambda: httpx.stream("POST", url, json=payload, timeout=self.timeout_s),
                _parse_ollama_line,
                on_chunk,  # type: ignore[arg-type]
            )

        try:
            if on_chunk is not None:
                text, truncated = retry_call(
                    attempt_stream, attempts=self.retries, sleep=self.sleep
                )
                data: dict[str, Any] = {"streamed": True}
            else:
                response = retry_call(attempt, attempts=self.retries, sleep=self.sleep)
                data = response.json()
                text = (data.get("message") or {}).get("content") or data.get("response") or ""
                truncated = data.get("done_reason") == "length"
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(self._error_message(exc, model)) from exc

        text = strip_reasoning(text)
        if not text.strip():
            raise LLMError(f"Ollama returned an empty response for {agent_id} using {model}.")
        return LLMResponse(text=text.strip(), model=model, raw=data, truncated=truncated)

    def _error_message(self, exc: Exception, model: str) -> str:
        """Explain a failed call. Only a connection failure suggests starting Ollama."""

        if _is_connection_failure(exc):
            return (
                f"Could not reach Ollama at {self.base_url}. Start it with `ollama serve` and "
                f"pull the model with `ollama pull {model}`. Provider error: {exc}"
            )
        if isinstance(exc, RetryableError):
            attempts = max(1, self.retries)
            return f"Ollama returned {exc} after {attempts} attempts for model {model}."
        if isinstance(exc, httpx.HTTPStatusError):
            pull = (
                f" Pull the model with `ollama pull {model}`."
                if exc.response.status_code == 404
                else ""
            )
            return (
                f"Ollama returned {_status_text(exc)} for model {model}.{pull} "
                f"Provider error: {exc}"
            )
        return f"Ollama request to {self.base_url} failed for model {model}. Provider error: {exc}"


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
        # Models whose schema form this server rejected; they get json_object straight away.
        self._json_object_models: set[str] = set()
        self._lock = threading.Lock()

    def _remember_json_object(self, model: str) -> None:
        with self._lock:
            if model in self._json_object_models:
                return
            self._json_object_models.add(model)
        logger.warning(
            "%s rejected the %s response format for %s; sending json_object from now on.",
            self.base_url,
            self.schema_mode,
            model,
        )

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

        def known_rejected() -> bool:
            return can_fall_back and model in self._json_object_models

        def attempt() -> httpx.Response:
            fell_back = known_rejected()
            try:
                response = httpx.post(
                    url,
                    json=fallback if fell_back else payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
                if (
                    can_fall_back
                    and not fell_back
                    and response.status_code in SCHEMA_REJECTED_STATUSES
                ):
                    fell_back = True
                    response = httpx.post(
                        url, json=fallback, headers=headers, timeout=self.timeout_s
                    )
            except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                raise RetryableError(str(exc)) from exc
            checked = _check_response(response)
            if fell_back:
                self._remember_json_object(model)
            return checked

        def stream_once(body: dict[str, Any], schema_requested: bool) -> tuple[str, bool]:
            return _consume_stream(
                "OpenAI-compatible provider",
                lambda: httpx.stream(
                    "POST", url, json=body, headers=headers, timeout=self.timeout_s
                ),
                _parse_sse_line,
                on_chunk,  # type: ignore[arg-type]
                SCHEMA_REJECTED_STATUSES if schema_requested else None,
            )

        def attempt_stream() -> tuple[str, bool]:
            if known_rejected():
                return stream_once(fallback, False)
            try:
                return stream_once(payload, can_fall_back)
            except _SchemaRejected:
                result = stream_once(fallback, False)
                self._remember_json_object(model)
                return result

        try:
            if on_chunk is not None:
                text, truncated = retry_call(
                    attempt_stream, attempts=self.retries, sleep=self.sleep
                )
                data: dict[str, Any] = {"streamed": True}
            else:
                response = retry_call(attempt, attempts=self.retries, sleep=self.sleep)
                data = response.json()
                choices = data.get("choices") or []
                first = choices[0] if choices else {}
                text = (first.get("message") or {}).get("content") or ""
                truncated = first.get("finish_reason") == "length"
        except LLMError:
            raise
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                detail = _status_text(exc)
            elif isinstance(exc, RetryableError) and not _is_connection_failure(exc):
                detail = f"{exc} after {max(1, self.retries)} attempts"
            else:
                detail = str(exc)
            raise LLMError(
                f"OpenAI-compatible provider failed at {self.base_url} with model {model}: {detail}"
            ) from exc

        text = strip_reasoning(text)
        if not text.strip():
            raise LLMError(
                f"OpenAI-compatible provider returned an empty response for {agent_id} "
                f"using {model}."
            )
        return LLMResponse(text=text.strip(), model=model, raw=data, truncated=truncated)


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
            text = json.dumps(demo_itinerary_payload(_demo_days(user)))
        else:
            text = json.dumps(demo_research_payload(agent_id, "your destination"))
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


def _demo_days(user: str) -> int:
    """The trip length the itinerary prompt asks for, as in "(8 days)"."""

    days_match = re.search(r"\((\d+)\s+days\)", user)
    return int(days_match.group(1)) if days_match else 8

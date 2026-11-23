"""Mock-based tests for LLM clients; httpx.post is patched, so nothing touches the network."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import httpx
import pytest
from httpx import HTTPStatusError, TimeoutException

from voyager_ai.config import NIM_BASE_URL, NIM_SCHEMA_MODE, Settings
from voyager_ai.llm import (
    DemoLLMClient,
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    apply_schema,
    build_llm_client,
)


class TestOllamaClient(unittest.TestCase):
    """Tests for OllamaClient with a mocked HTTP transport."""

    def setUp(self) -> None:
        self.client = OllamaClient(base_url="http://localhost:11434", timeout_s=10)
        self.mock_httpx = MagicMock()
        self.post_patch = patch("voyager_ai.llm.httpx.post", self.mock_httpx.post)
        self.post_patch.start()
        self.addCleanup(self.post_patch.stop)

    def _mock_response(self, json_data: dict, status: int = 200) -> MagicMock:
        response = MagicMock()
        response.json.return_value = json_data
        response.status_code = status
        response.raise_for_status.return_value = None
        self.mock_httpx.post.return_value = response
        return response

    def test_successful_completion(self) -> None:
        self._mock_response(
            {
                "message": {"content": "Hello from Ollama!"},
                "model": "qwen3:8b",
            }
        )
        result = self.client.complete(
            agent_id="flight",
            model="qwen3:8b",
            system="You are a helpful assistant.",
            user="Suggest a destination.",
        )
        self.assertEqual(result.text, "Hello from Ollama!")
        self.assertEqual(result.model, "qwen3:8b")

        call_kwargs = self.mock_httpx.post.call_args
        url = call_kwargs[0][0]
        self.assertIn("/api/chat", url)
        payload = call_kwargs[1]["json"]
        self.assertEqual(payload["model"], "qwen3:8b")

    def test_successful_completion_with_schema(self) -> None:
        self._mock_response(
            {
                "message": {"content": '{"destinations": [{"name": "Tokyo"}]}'},
                "model": "qwen3:8b",
            }
        )
        schema = {"type": "object", "properties": {"destinations": {"type": "array"}}}
        result = self.client.complete(
            agent_id="discovery",
            model="qwen3:8b",
            system="",
            user="Suggest cities.",
            json_schema=schema,
        )
        self.assertIn("Tokyo", result.text)
        payload = self.mock_httpx.post.call_args[1]["json"]
        self.assertIn("format", payload)

    def test_http_error_raises_llm_error(self) -> None:
        response = MagicMock()
        response.raise_for_status.side_effect = HTTPStatusError(
            "500 Server Error", request=MagicMock(), response=MagicMock()
        )
        self.mock_httpx.post.return_value = response

        with self.assertRaises(LLMError) as ctx:
            self.client.complete(
                agent_id="visa",
                model="phi4-mini",
                system="",
                user="Visa info.",
            )
        self.assertIn("Provider error", str(ctx.exception))

    def test_empty_response_raises_llm_error(self) -> None:
        self._mock_response({"message": {"content": "   "}})
        with self.assertRaises(LLMError) as ctx:
            self.client.complete(
                agent_id="hotel",
                model="qwen3:8b",
                system="",
                user="Hotels?",
            )
        self.assertIn("empty response", str(ctx.exception).lower())

    def test_rollback_to_legacy_response_key(self) -> None:
        """Some older Ollama models use 'response' key at root level."""
        self._mock_response({"response": "Fallback text", "model": "phi4-mini"})
        result = self.client.complete(
            agent_id="visa",
            model="phi4-mini",
            system="",
            user="Visa rules.",
        )
        self.assertEqual(result.text, "Fallback text")

    def test_num_ctx_sent_in_options(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", timeout_s=10, num_ctx=8192)
        self._mock_response({"message": {"content": "ok"}})
        client.complete(agent_id="itinerary", model="m", system="", user="x")
        options = self.mock_httpx.post.call_args[1]["json"]["options"]
        self.assertEqual(options["num_ctx"], 8192)

    def test_timeout_is_passed_to_httpx(self) -> None:
        """Verify that timeout_s is forwarded to the httpx call."""
        client = OllamaClient(base_url="http://localhost:11434", timeout_s=42)
        self._mock_response({"message": {"content": "ok"}, "model": "m"})
        client.complete(agent_id="itinerary", model="m", system="", user="Plan trip.")
        self.assertEqual(self.mock_httpx.post.call_args[1]["timeout"], 42)

    def _resp(self, status: int, json_data: dict | None = None, headers: dict | None = None):
        response = MagicMock()
        response.status_code = status
        response.headers = headers or {}
        response.json.return_value = json_data or {}
        return response

    def test_ollama_retries_429_using_retry_after(self) -> None:
        sleeps: list[float] = []
        client = OllamaClient(timeout_s=10, sleep=sleeps.append)
        self.mock_httpx.post.side_effect = [
            self._resp(429, headers={"Retry-After": "3"}),
            self._resp(200, {"message": {"content": "ok"}}),
        ]
        result = client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertEqual(result.text, "ok")
        self.assertEqual(sleeps, [3.0])
        self.assertEqual(self.mock_httpx.post.call_count, 2)

    def test_ollama_does_not_retry_400(self) -> None:
        response = self._resp(400)
        response.raise_for_status.side_effect = HTTPStatusError(
            "400 Bad Request", request=MagicMock(), response=response
        )
        self.mock_httpx.post.return_value = response
        client = OllamaClient(timeout_s=10, sleep=lambda s: None)
        with self.assertRaises(LLMError):
            client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertEqual(self.mock_httpx.post.call_count, 1)

    def test_ollama_retries_connect_error_then_raises_llm_error(self) -> None:
        client = OllamaClient(timeout_s=10, retries=2, sleep=lambda s: None)
        self.mock_httpx.post.side_effect = httpx.ConnectError("refused")
        with self.assertRaises(LLMError) as ctx:
            client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertIn("Could not reach Ollama", str(ctx.exception))
        self.assertEqual(self.mock_httpx.post.call_count, 2)

    def test_ollama_does_not_retry_read_timeout(self) -> None:
        client = OllamaClient(timeout_s=10, sleep=lambda s: None)
        self.mock_httpx.post.side_effect = httpx.ReadTimeout("slow")
        with self.assertRaises(LLMError):
            client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertEqual(self.mock_httpx.post.call_count, 1)

    def test_ollama_sends_think_false_by_default(self) -> None:
        self._mock_response({"message": {"content": "ok"}})
        self.client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertIs(self.mock_httpx.post.call_args[1]["json"]["think"], False)
        for think, expected in (("true", True),):
            client = OllamaClient(timeout_s=10, think=think)
            client.complete(agent_id="visa", model="m", system="", user="x")
            self.assertIs(self.mock_httpx.post.call_args[1]["json"]["think"], expected)
        client = OllamaClient(timeout_s=10, think="default")
        client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertNotIn("think", self.mock_httpx.post.call_args[1]["json"])

    def test_reasoning_is_stripped_from_content(self) -> None:
        self._mock_response({"message": {"content": '<think>plan</think>\n{"a": 1}'}})
        result = self.client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertEqual(result.text, '{"a": 1}')


class TestOpenAICompatibleClient(unittest.TestCase):
    """Tests for OpenAICompatibleClient with a mocked HTTP transport."""

    def setUp(self) -> None:
        self.client = OpenAICompatibleClient(
            base_url="http://localhost:1234/v1",
            api_key="test-key",
            timeout_s=10,
        )
        self.mock_httpx = MagicMock()
        self.post_patch = patch("voyager_ai.llm.httpx.post", self.mock_httpx.post)
        self.post_patch.start()
        self.addCleanup(self.post_patch.stop)

    def _mock_response(self, json_data: dict) -> MagicMock:
        response = MagicMock()
        response.json.return_value = json_data
        response.status_code = 200
        response.raise_for_status.return_value = None
        self.mock_httpx.post.return_value = response
        return response

    def test_successful_completion(self) -> None:
        self._mock_response({"choices": [{"message": {"content": "Hello from local server!"}}]})
        result = self.client.complete(
            agent_id="experience",
            model="qwen3:8b",
            system="You are an expert.",
            user="Suggest experiences.",
        )
        self.assertEqual(result.text, "Hello from local server!")

        call_kwargs = self.mock_httpx.post.call_args
        url = call_kwargs[0][0]
        self.assertIn("/chat/completions", url)
        headers = call_kwargs[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    def test_empty_response_raises_llm_error(self) -> None:
        self._mock_response({"choices": []})
        with self.assertRaises(LLMError) as ctx:
            self.client.complete(
                agent_id="visa",
                model="qwen3:8b",
                system="",
                user="Visa?",
            )
        self.assertIn("empty response", str(ctx.exception).lower())

    def test_http_error_raises_llm_error(self) -> None:
        response = MagicMock()
        response.raise_for_status.side_effect = TimeoutException("Request timed out")
        self.mock_httpx.post.return_value = response

        with self.assertRaises(LLMError):
            self.client.complete(
                agent_id="flight",
                model="qwen3:8b",
                system="",
                user="Flights?",
            )

    def test_schema_is_sent_as_json_schema(self) -> None:
        self._mock_response({"choices": [{"message": {"content": "{}"}}]})
        schema = {"type": "object"}
        self.client.complete(
            agent_id="discovery", model="m", system="", user="x", json_schema=schema
        )
        sent = self.mock_httpx.post.call_args[1]["json"]["response_format"]
        self.assertEqual(sent["type"], "json_schema")
        self.assertEqual(sent["json_schema"]["schema"], schema)
        self.assertEqual(self.mock_httpx.post.call_count, 1)

    def test_falls_back_to_json_mode_when_schema_is_rejected(self) -> None:
        rejected = MagicMock(status_code=400)
        accepted = MagicMock(status_code=200)
        accepted.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
        self.mock_httpx.post.side_effect = [rejected, accepted]
        result = self.client.complete(
            agent_id="discovery", model="m", system="", user="x", json_schema={"type": "object"}
        )
        self.assertEqual(result.text, "{}")
        calls = self.mock_httpx.post.call_args_list
        self.assertEqual(calls[0][1]["json"]["response_format"]["type"], "json_schema")
        self.assertEqual(calls[1][1]["json"]["response_format"], {"type": "json_object"})

    def test_no_response_format_without_schema(self) -> None:
        self._mock_response({"choices": [{"message": {"content": "ok"}}]})
        self.client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertNotIn("response_format", self.mock_httpx.post.call_args[1]["json"])

    def test_openai_compatible_retries_503_then_succeeds(self) -> None:
        client = OpenAICompatibleClient("http://localhost:1234/v1", sleep=lambda s: None)
        busy = MagicMock(status_code=503, headers={})
        ok = MagicMock(status_code=200)
        ok.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        self.mock_httpx.post.side_effect = [busy, ok]
        result = client.complete(agent_id="visa", model="m", system="", user="x")
        self.assertEqual(result.text, "ok")
        self.assertEqual(self.mock_httpx.post.call_count, 2)


class TestBuildLLMClient(unittest.TestCase):
    """Tests for the build_llm_client factory function."""

    def test_build_ollama(self) -> None:
        settings = Settings(provider="ollama")
        client = build_llm_client(settings)
        from voyager_ai.llm import OllamaClient

        self.assertIsInstance(client, OllamaClient)

    def test_build_openai_compatible(self) -> None:
        settings = Settings(provider="openai-compatible")
        client = build_llm_client(settings)
        from voyager_ai.llm import OpenAICompatibleClient

        self.assertIsInstance(client, OpenAICompatibleClient)

    def test_build_openai_variants(self) -> None:
        for variant in ("openai_compatible", "openai"):
            settings = Settings(provider=variant)
            client = build_llm_client(settings)
            from voyager_ai.llm import OpenAICompatibleClient

            self.assertIsInstance(client, OpenAICompatibleClient)

    def test_build_demo(self) -> None:
        settings = Settings(provider="demo")
        client = build_llm_client(settings)
        from voyager_ai.llm import DemoLLMClient

        self.assertIsInstance(client, DemoLLMClient)

    def test_build_unsupported_provider_raises_error(self) -> None:
        settings = Settings(provider="unknown-provider")
        with self.assertRaises(LLMError):
            build_llm_client(settings)


def _stream_ctx(lines, status: int = 200, error: Exception | None = None) -> MagicMock:
    """A context manager standing in for ``httpx.stream(...)``."""

    response = MagicMock()
    response.status_code = status
    response.headers = {}
    response.raise_for_status.return_value = None

    def iter_lines():
        yield from lines
        if error is not None:
            raise error

    response.iter_lines.side_effect = iter_lines
    ctx = MagicMock()
    ctx.__enter__.return_value = response
    ctx.__exit__.return_value = False
    return ctx


def _no_sleep(_seconds: float) -> None:
    return None


class TestStreaming(unittest.TestCase):
    def _patch_stream(self, *contexts: MagicMock) -> MagicMock:
        patcher = patch("voyager_ai.llm.httpx.stream", side_effect=list(contexts))
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def test_ollama_streams_chunks(self) -> None:
        stream = self._patch_stream(
            _stream_ctx(
                [
                    '{"message":{"content":"Hel"},"done":false}',
                    '{"message":{"content":"lo"},"done":true}',
                ]
            )
        )
        chunks: list[str] = []
        result = OllamaClient(sleep=_no_sleep).complete(
            agent_id="a", model="m", system="s", user="u", on_chunk=chunks.append
        )
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertEqual(result.text, "Hello")
        self.assertTrue(stream.call_args[1]["json"]["stream"])

    def test_openai_streams_sse_chunks(self) -> None:
        self._patch_stream(
            _stream_ctx(
                [
                    ": keepalive",
                    'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                    "",
                    'data: {"choices":[{"delta":{"content":"lo"}}]}',
                    "data: [DONE]",
                ]
            )
        )
        chunks: list[str] = []
        result = OpenAICompatibleClient("http://x/v1", sleep=_no_sleep).complete(
            agent_id="a", model="m", system="s", user="u", on_chunk=chunks.append
        )
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertEqual(result.text, "Hello")

    def test_stream_interrupted_after_first_chunk_is_not_retried(self) -> None:
        stream = self._patch_stream(
            _stream_ctx(
                ['data: {"choices":[{"delta":{"content":"Hel"}}]}'],
                error=httpx.ReadError("reset"),
            )
        )
        with self.assertRaises(LLMError) as ctx:
            OpenAICompatibleClient("http://x/v1", sleep=_no_sleep).complete(
                agent_id="a", model="m", system="s", user="u", on_chunk=lambda _c: None
            )
        self.assertIn("interrupted after 3 characters", str(ctx.exception))
        self.assertEqual(stream.call_count, 1)

    def test_stream_503_before_any_chunk_is_retried(self) -> None:
        stream = self._patch_stream(
            _stream_ctx([], status=503),
            _stream_ctx(['{"message":{"content":"ok"},"done":true}']),
        )
        result = OllamaClient(sleep=_no_sleep).complete(
            agent_id="a", model="m", system="s", user="u", on_chunk=lambda _c: None
        )
        self.assertEqual(result.text, "ok")
        self.assertEqual(stream.call_count, 2)

    def test_openai_stream_schema_rejected_falls_back_to_json_object(self) -> None:
        stream = self._patch_stream(
            _stream_ctx([], status=400),
            _stream_ctx(['data: {"choices":[{"delta":{"content":"{}"}}]}', "data: [DONE]"]),
        )
        result = OpenAICompatibleClient("http://x/v1", sleep=_no_sleep).complete(
            agent_id="a",
            model="m",
            system="s",
            user="u",
            json_schema={"type": "object"},
            on_chunk=lambda _c: None,
        )
        self.assertEqual(result.text, "{}")
        self.assertEqual(stream.call_count, 2)
        first, second = (c[1]["json"] for c in stream.call_args_list)
        self.assertEqual(first["response_format"]["type"], "json_schema")
        self.assertEqual(second["response_format"], {"type": "json_object"})
        self.assertTrue(second["stream"])

    def test_demo_client_streams_three_pieces(self) -> None:
        chunks: list[str] = []
        result = DemoLLMClient().complete(
            agent_id="flight", model="demo", system="", user="u", on_chunk=chunks.append
        )
        self.assertEqual(len(chunks), 3)
        self.assertEqual("".join(chunks), result.text)


def test_apply_schema_modes():
    base, schema = {"model": "m"}, {"type": "object"}
    assert apply_schema(base, schema, "json_schema", "d")["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "d", "schema": schema},
    }
    assert apply_schema(base, schema, "nvext_guided", "d")["nvext"] == {"guided_json": schema}
    assert apply_schema(base, schema, "vllm_guided", "d")["guided_json"] == schema
    assert apply_schema(base, schema, "json_object", "d")["response_format"] == {
        "type": "json_object"
    }
    assert base == {"model": "m"}


def _ok_response() -> MagicMock:
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"choices": [{"message": {"content": "{}"}}]}
    return ok


def test_nvext_mode_falls_back_to_json_object_on_400():
    client = OpenAICompatibleClient("http://x/v1", schema_mode="nvext_guided", sleep=_no_sleep)
    with patch(
        "voyager_ai.llm.httpx.post", side_effect=[MagicMock(status_code=400), _ok_response()]
    ) as post:
        client.complete(
            agent_id="a", model="m", system="", user="u", json_schema={"type": "object"}
        )
    first, second = (c[1]["json"] for c in post.call_args_list)
    assert first["nvext"] == {"guided_json": {"type": "object"}}
    assert second["response_format"] == {"type": "json_object"}
    assert "nvext" not in second


def test_fallback_disabled_does_not_retry_400():
    client = OpenAICompatibleClient("http://x/v1", fallback=False, sleep=_no_sleep)
    rejected = MagicMock(status_code=400)
    rejected.raise_for_status.side_effect = HTTPStatusError(
        "400", request=MagicMock(), response=rejected
    )
    with patch("voyager_ai.llm.httpx.post", return_value=rejected) as post:
        with pytest.raises(LLMError):
            client.complete(
                agent_id="a", model="m", system="", user="u", json_schema={"type": "object"}
            )
    assert post.call_count == 1


def test_json_object_mode_has_no_fallback_request():
    client = OpenAICompatibleClient("http://x/v1", schema_mode="json_object", sleep=_no_sleep)
    rejected = MagicMock(status_code=400)
    rejected.raise_for_status.side_effect = HTTPStatusError(
        "400", request=MagicMock(), response=rejected
    )
    with patch("voyager_ai.llm.httpx.post", return_value=rejected) as post:
        with pytest.raises(LLMError):
            client.complete(
                agent_id="a", model="m", system="", user="u", json_schema={"type": "object"}
            )
    assert post.call_count == 1


def test_build_nim_requires_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        build_llm_client(Settings(provider="nim"))


def test_build_nim_client():
    client = build_llm_client(Settings(provider="nim", nvidia_api_key="k"))
    assert isinstance(client, OpenAICompatibleClient)
    assert client.base_url == NIM_BASE_URL
    assert client.api_key == "k"
    assert client.schema_mode == NIM_SCHEMA_MODE


def test_schema_mode_setting_overrides_provider_default():
    client = build_llm_client(
        Settings(provider="nim", nvidia_api_key="k", schema_mode="nvext_guided")
    )
    assert client.schema_mode == "nvext_guided"
    assert build_llm_client(Settings(provider="openai")).schema_mode == "json_schema"

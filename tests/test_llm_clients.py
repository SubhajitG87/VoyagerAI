"""Mock-based integration tests for LLM clients using sys.modules patching."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

# Import real httpx exceptions before sys.modules patching intercepts httpx.
try:
    from httpx import HTTPStatusError, TimeoutException
    _HAVE_HTTPX = True
except ImportError:
    HTTPStatusError = Exception
    TimeoutException = Exception
    _HAVE_HTTPX = False

from voyager_ai.config import Settings
from voyager_ai.llm import (
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    build_llm_client,
)


class TestOllamaClient(unittest.TestCase):
    """Tests for OllamaClient with a mocked HTTP transport."""

    def setUp(self) -> None:
        self.client = OllamaClient(base_url="http://localhost:11434", timeout_s=10)
        self.mock_httpx = MagicMock()
        self.modules_patch = patch.dict("sys.modules", {"httpx": self.mock_httpx})
        self.modules_patch.start()

    def tearDown(self) -> None:
        self.modules_patch.stop()

    def _mock_response(self, json_data: dict, status: int = 200) -> MagicMock:
        response = MagicMock()
        response.json.return_value = json_data
        response.status_code = status
        response.raise_for_status.return_value = None
        self.mock_httpx.post.return_value = response
        return response

    def test_successful_completion(self) -> None:
        self._mock_response({
            "message": {"content": "Hello from Ollama!"},
            "model": "qwen3:8b",
        })
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
        self._mock_response({
            "message": {"content": '{"destinations": [{"name": "Tokyo"}]}'},
            "model": "qwen3:8b",
        })
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

    @unittest.skipIf(not _HAVE_HTTPX, "httpx is not installed")
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

    def test_timeout_is_passed_to_httpx(self) -> None:
        """Verify that timeout_s is forwarded to the httpx call."""
        client = OllamaClient(base_url="http://localhost:11434", timeout_s=42)
        mock_httpx = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            mock_response = MagicMock()
            mock_response.json.return_value = {"message": {"content": "ok"}, "model": "m"}
            mock_httpx.post.return_value = mock_response
            client.complete(
                agent_id="itinerary", model="m", system="", user="Plan trip."
            )
            self.assertEqual(mock_httpx.post.call_args[1]["timeout"], 42)


class TestOpenAICompatibleClient(unittest.TestCase):
    """Tests for OpenAICompatibleClient with a mocked HTTP transport."""

    def setUp(self) -> None:
        self.client = OpenAICompatibleClient(
            base_url="http://localhost:1234/v1",
            api_key="test-key",
            timeout_s=10,
        )
        self.mock_httpx = MagicMock()
        self.modules_patch = patch.dict("sys.modules", {"httpx": self.mock_httpx})
        self.modules_patch.start()

    def tearDown(self) -> None:
        self.modules_patch.stop()

    def _mock_response(self, json_data: dict) -> MagicMock:
        response = MagicMock()
        response.json.return_value = json_data
        response.status_code = 200
        response.raise_for_status.return_value = None
        self.mock_httpx.post.return_value = response
        return response

    def test_successful_completion(self) -> None:
        self._mock_response({
            "choices": [{"message": {"content": "Hello from local server!"}}]
        })
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

    @unittest.skipIf(not _HAVE_HTTPX, "httpx is not installed")
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


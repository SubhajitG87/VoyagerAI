"""Mock-based tests for the search client wrappers."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

from voyager_ai.config import Settings
from voyager_ai.models import Destination, SearchResult, TravelPreferences
from voyager_ai.search import (
    DuckDuckGoSearchClient,
    NoopSearchClient,
    build_search_client,
    format_search_context,
    queries_for_agent,
)


class TestNoopSearchClient(unittest.TestCase):
    """The NoopSearchClient always returns empty results."""

    def test_search_returns_empty(self) -> None:
        client = NoopSearchClient()
        self.assertEqual(client.search("any query"), [])

    def test_search_many_deduplicates_nothing(self) -> None:
        client = NoopSearchClient()
        self.assertEqual(client.search_many(["q1", "q2"]), [])


class TestDuckDuckGoSearchClient(unittest.TestCase):
    """Tests for DuckDuckGoSearchClient with sys.modules DDGS patched."""

    def setUp(self) -> None:
        self.patchers = []

    def tearDown(self) -> None:
        for p in self.patchers:
            p.stop()

    def _patch_ddgs(self, raw_results):
        mock_ddgs_module = MagicMock()
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = raw_results or []
        mock_ddgs_module.DDGS.return_value.__enter__.return_value = mock_ddgs_instance
        p = patch.dict("sys.modules", {"ddgs": mock_ddgs_module})
        p.start()
        self.patchers.append(p)
        return mock_ddgs_module

    def test_search_returns_results(self) -> None:
        self._patch_ddgs([
            {
                "title": "Tokyo Travel Guide",
                "href": "https://example.com/tokyo",
                "body": "A comprehensive travel guide for Tokyo.",
            },
            {
                "title": "Best Hotels in Tokyo",
                "href": "https://example.com/hotels",
                "body": "Top-rated hotels in Tokyo for every budget.",
            },
        ])
        client = DuckDuckGoSearchClient()
        results = client.search("Tokyo travel", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].title, "Tokyo Travel Guide")
        self.assertEqual(results[0].url, "https://example.com/tokyo")

    def test_search_returns_empty_when_no_title(self) -> None:
        self._patch_ddgs([
            {"title": "", "href": "", "body": "Some content."},
        ])
        client = DuckDuckGoSearchClient()
        results = client.search("anything")
        self.assertEqual(len(results), 0)

    def test_search_handles_api_error_gracefully(self) -> None:
        mock_mod = MagicMock()
        mock_mod.DDGS.side_effect = RuntimeError("API unavailable")
        p = patch.dict("sys.modules", {"ddgs": mock_mod})
        p.start()
        self.patchers.append(p)

        client = DuckDuckGoSearchClient()
        results = client.search("any query")
        self.assertEqual(results, [])

    def test_search_handles_invalid_item_keys(self) -> None:
        self._patch_ddgs([
            {
                "heading": "Fallback Title",
                "url": "https://example.com/fallback",
                "snippet": "Fallback snippet.",
            },
        ])
        client = DuckDuckGoSearchClient()
        results = client.search("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Fallback Title")

    def test_missing_ddgs_package_returns_empty(self) -> None:
        old = sys.modules.pop("ddgs", None)
        try:
            client = DuckDuckGoSearchClient()
            results = client.search("anything")
            self.assertEqual(results, [])
        finally:
            if old is not None:
                sys.modules["ddgs"] = old

class TestBuildSearchClient(unittest.TestCase):
    def test_enabled_duckduckgo(self) -> None:
        settings = Settings(search_enabled=True, search_provider="duckduckgo")
        client = build_search_client(settings)
        self.assertIsInstance(client, DuckDuckGoSearchClient)

    def test_disabled_search_returns_noop(self) -> None:
        settings = Settings(search_enabled=False)
        client = build_search_client(settings)
        self.assertIsInstance(client, NoopSearchClient)

    def test_disabled_provider_values(self) -> None:
        for provider in ("none", "off", "disabled"):
            settings = Settings(search_enabled=True, search_provider=provider)
            client = build_search_client(settings)
            self.assertIsInstance(client, NoopSearchClient)

    def test_unknown_provider_falls_back_to_noop(self) -> None:
        settings = Settings(search_enabled=True, search_provider="unknown")
        client = build_search_client(settings)
        self.assertIsInstance(client, NoopSearchClient)


class TestQueriesForAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.prefs = TravelPreferences(
            departure_city="Mumbai",
            nationality="Indian",
            vibe="food",
            months=6,
            nights=7,
        )
        self.dest = Destination(name="Tokyo", country="Japan", why="Food paradise")

    def test_flight_queries(self) -> None:
        queries = queries_for_agent("flight", self.prefs, self.dest)
        self.assertEqual(len(queries), 3)
        self.assertTrue(any("Mumbai" in q for q in queries))
        self.assertTrue(any("Tokyo" in q for q in queries))

    def test_visa_queries(self) -> None:
        queries = queries_for_agent("visa", self.prefs, self.dest)
        self.assertTrue(any("Indian" in q for q in queries))
        self.assertTrue(any("Japan" in q for q in queries))

    def test_hotel_queries(self) -> None:
        queries = queries_for_agent("hotel", self.prefs, self.dest)
        self.assertTrue(any("Tokyo" in q for q in queries))

    def test_experience_queries(self) -> None:
        queries = queries_for_agent("experience", self.prefs, self.dest)
        self.assertTrue(any("food" in q.lower() for q in queries))
        self.assertTrue(any("Tokyo" in q for q in queries))

    def test_unknown_agent_returns_empty(self) -> None:
        queries = queries_for_agent("unknown_agent", self.prefs, self.dest)
        self.assertEqual(queries, [])


class TestFormatSearchContext(unittest.TestCase):
    def test_no_results_returns_default_message(self) -> None:
        result = format_search_context([])
        self.assertIn("No live web snippets", result)

    def test_with_results_returns_formatted_markdown(self) -> None:
        results = [
            SearchResult(title="Guide", url="https://example.com", snippet="A guide."),
        ]
        result = format_search_context(results)
        self.assertIn("Use these search snippets", result)
        self.assertIn("Guide", result)
        self.assertIn("https://example.com", result)
        self.assertIn("A guide.", result)

    def test_limit_is_respected(self) -> None:
        results = [
            SearchResult(title=f"Result {i}", url=f"https://example.com/{i}")
            for i in range(20)
        ]
        result = format_search_context(results, limit=5)
        self.assertEqual(len(result.splitlines()), 6)

    def test_snippet_field_is_optional(self) -> None:
        results = [SearchResult(title="No Snippet", url="https://example.com")]
        result = format_search_context(results)
        self.assertIn("No Snippet", result)


"""Mock-based tests for the search client wrappers."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

from voyager_ai.config import Settings
from voyager_ai.fetch import Page
from voyager_ai.models import Destination, SearchResult, TravelPreferences
from voyager_ai.search import (
    DuckDuckGoSearchClient,
    NoopSearchClient,
    SearchClient,
    SearchError,
    build_search_client,
    format_search_context,
    is_official_domain,
    queries_for_agent,
    rank_results,
)


class TestNoopSearchClient(unittest.TestCase):
    """The NoopSearchClient always returns empty results."""

    def test_search_returns_empty(self) -> None:
        client = NoopSearchClient()
        self.assertEqual(client.search("any query"), [])

    def test_search_many_deduplicates_nothing(self) -> None:
        client = NoopSearchClient()
        outcome = client.search_many(["q1", "q2"])
        self.assertEqual(outcome.results, [])
        self.assertEqual(outcome.errors, [])


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
        self._patch_ddgs(
            [
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
            ]
        )
        client = DuckDuckGoSearchClient()
        results = client.search("Tokyo travel", max_results=2)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].title, "Tokyo Travel Guide")
        self.assertEqual(results[0].url, "https://example.com/tokyo")

    def test_search_returns_empty_when_no_title(self) -> None:
        self._patch_ddgs(
            [
                {"title": "", "href": "", "body": "Some content."},
            ]
        )
        client = DuckDuckGoSearchClient()
        results = client.search("anything")
        self.assertEqual(len(results), 0)

    def test_search_handles_api_error_gracefully(self) -> None:
        mock_mod = MagicMock()
        mock_mod.DDGS.side_effect = RuntimeError("API unavailable")
        p = patch.dict("sys.modules", {"ddgs": mock_mod})
        p.start()
        self.patchers.append(p)

        client = DuckDuckGoSearchClient(retries=1)
        with self.assertRaises(SearchError):
            client.search("any query")

    def test_search_handles_invalid_item_keys(self) -> None:
        self._patch_ddgs(
            [
                {
                    "heading": "Fallback Title",
                    "url": "https://example.com/fallback",
                    "snippet": "Fallback snippet.",
                },
            ]
        )
        client = DuckDuckGoSearchClient()
        results = client.search("test")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Fallback Title")

    def test_imports_ddgs_when_not_preloaded(self) -> None:
        mock_module = MagicMock()
        mock_module.DDGS.return_value.__enter__.return_value.text.return_value = [
            {"title": "T", "href": "https://example.com/t", "body": "B"}
        ]
        with patch.dict("sys.modules"):
            sys.modules.pop("ddgs", None)
            sys.modules.pop("duckduckgo_search", None)
            with patch(
                "voyager_ai.search.importlib.import_module", return_value=mock_module
            ) as importer:
                results = DuckDuckGoSearchClient().search("q")
        importer.assert_called_once_with("ddgs")
        self.assertEqual([r.url for r in results], ["https://example.com/t"])

    def test_search_failure_raises_search_error_with_reason(self) -> None:
        module = self._patch_ddgs([])
        module.DDGS.return_value.__enter__.return_value.text.side_effect = RuntimeError(
            "rate limited"
        )
        with self.assertRaises(SearchError) as ctx:
            DuckDuckGoSearchClient(retries=1).search("q")
        self.assertIn("rate limited", str(ctx.exception))


class TestBuildSearchClient(unittest.TestCase):
    def test_enabled_duckduckgo(self) -> None:
        settings = Settings(search_enabled=True, search_provider="duckduckgo")
        client = build_search_client(settings)
        self.assertIsInstance(client, DuckDuckGoSearchClient)

    def test_demo_provider_never_searches(self) -> None:
        settings = Settings(provider="demo", search_enabled=True, search_provider="duckduckgo")
        self.assertIsInstance(build_search_client(settings), NoopSearchClient)

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
        self.assertIn("Snippet: A guide.", result)
        self.assertIn("Guide", result)
        self.assertIn("https://example.com", result)
        self.assertIn("A guide.", result)

    def test_limit_is_respected(self) -> None:
        results = [
            SearchResult(title=f"Result {i}", url=f"https://example.com/{i}") for i in range(20)
        ]
        result = format_search_context(results, limit=5)
        self.assertEqual(result.count("Source ["), 5)

    def test_snippet_field_is_optional(self) -> None:
        results = [SearchResult(title="No Snippet", url="https://example.com")]
        result = format_search_context(results)
        self.assertIn("No Snippet", result)


def r(name: str) -> SearchResult:
    return SearchResult(name, f"https://{name}", "")


def r_url(url: str) -> SearchResult:
    return SearchResult("t", url, "")


class FakeSearch(SearchClient):
    def __init__(self, table):
        self.table = table

    def search(self, query, max_results=4):
        value = self.table[query]
        if isinstance(value, Exception):
            raise value
        return value


def test_search_many_records_failures_and_keeps_other_results():
    client = FakeSearch({"q1": [r("a")], "q2": SearchError("rate limited"), "q3": [r("b")]})
    outcome = client.search_many(["q1", "q2", "q3"])
    assert [x.url for x in outcome.results] == ["https://a", "https://b"]
    assert outcome.errors == ["Search failed for 'q2': rate limited"]


def test_ddg_retries_then_succeeds(monkeypatch):
    calls = []

    class FlakyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results=4):
            calls.append(query)
            if len(calls) == 1:
                raise RuntimeError("blip")
            return [{"title": "T", "href": "https://x.test/t", "body": "b"}]

    module = MagicMock()
    module.DDGS = FlakyDDGS
    monkeypatch.setitem(sys.modules, "ddgs", module)
    sleeps: list[float] = []
    client = DuckDuckGoSearchClient(retries=2, sleep=sleeps.append)
    results = client.search("q")
    assert len(results) == 1
    assert len(sleeps) == 1


def test_missing_ddgs_is_an_error(monkeypatch):
    monkeypatch.setattr("voyager_ai.search._load_ddgs_module", lambda: None)
    with pytest.raises(SearchError, match="pip install ddgs"):
        DuckDuckGoSearchClient().search("q")


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.gov.uk/foreign-travel-advice", True),
        ("https://www.mofa.go.jp/j_info/visit/visa/", True),
        ("https://travel.state.gov/content/travel.html", True),
        ("https://www.canada.ca/en/immigration", True),
        ("https://indianvisaonline.gov.in/", True),
        ("https://govisa.example.com/", False),
        ("https://visa-gov.com/", False),
        ("https://go.com/", False),
    ],
)
def test_is_official_domain(url, expected):
    assert is_official_domain(url) is expected


def test_visa_ranking_puts_official_first_and_is_stable():
    ranked = rank_results("visa", [r("blog1"), r_url("https://www.gov.uk/x"), r("blog2")])
    assert [x.url for x in ranked] == ["https://www.gov.uk/x", "https://blog1", "https://blog2"]
    assert rank_results("hotel", ranked[::-1]) == ranked[::-1]


def test_build_search_client_passes_retries():
    client = build_search_client(Settings(search_retries=5))
    assert isinstance(client, DuckDuckGoSearchClient)
    assert client.retries == 5


def test_context_lists_pages_then_snippets_with_cite_rule() -> None:
    ctx = format_search_context([r("a"), r("b")], pages=[Page("https://a", "A", "Body A")])
    assert ctx.index("Body A") < ctx.index("https://b")
    assert ctx.count("https://a") == 1
    assert "Source [1]: A\nURL: https://a\nBody A" in ctx
    assert "Source [2]: b\nURL: https://b\nSnippet: " in ctx
    assert ctx.rstrip().endswith('set basis to "model" and source_url to null.')


def test_context_limit_counts_pages_and_snippets() -> None:
    results = [r_url(f"https://x/{i}") for i in range(10)]
    pages = [Page(f"https://p/{i}", "P", "body") for i in range(2)]
    assert format_search_context(results, pages=pages, limit=5).count("Source [") == 5

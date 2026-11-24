"""Optional public web search snippets for the specialist agents."""

from __future__ import annotations

import importlib
import logging
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from urllib.parse import urlparse

from voyager_ai.config import Settings
from voyager_ai.fetch import Page
from voyager_ai.models import (
    Destination,
    SearchResult,
    TravelPreferences,
    VoyagerError,
)
from voyager_ai.retry import RetryableError, retry_call

logger = logging.getLogger(__name__)


def _load_ddgs_module() -> ModuleType | None:
    """Import the search library, preferring the current package name."""

    for module_name in ("ddgs", "duckduckgo_search"):
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


class SearchError(VoyagerError):
    """A web search could not be completed."""


@dataclass
class SearchOutcome:
    results: list[SearchResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SearchClient:
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        raise NotImplementedError

    def search_many(self, queries: Iterable[str], max_results: int = 4) -> SearchOutcome:
        seen = set()
        outcome = SearchOutcome()
        for query in queries:
            try:
                found = self.search(query, max_results=max_results)
            except SearchError as exc:
                outcome.errors.append(f"Search failed for {query!r}: {exc}")
                continue
            for result in found:
                key = result.url.lower()
                if key in seen:
                    continue
                seen.add(key)
                outcome.results.append(result)
        return outcome


class NoopSearchClient(SearchClient):
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        return []


class DuckDuckGoSearchClient(SearchClient):
    """Tiny wrapper around the optional ddgs / duckduckgo-search package."""

    def __init__(self, retries: int = 2, sleep: Callable[[float], None] = time.sleep):
        self.retries = retries
        self.sleep = sleep

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        ddgs_module = _load_ddgs_module()
        if ddgs_module is None:
            raise SearchError("The ddgs package is not installed; run pip install ddgs.")

        def attempt() -> list[dict]:
            try:
                with ddgs_module.DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=max_results))
            except Exception as exc:
                raise RetryableError(str(exc) or type(exc).__name__) from exc

        try:
            raw_results = retry_call(attempt, attempts=self.retries, sleep=self.sleep)
        except RetryableError as exc:
            raise SearchError(str(exc)) from exc

        results = []
        for item in raw_results:
            title = str(item.get("title") or item.get("heading") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results


_GOV_HOST = re.compile(r"(^|\.)(gov|gouv|gob|govt|go|gc)\.[a-z]{2}$")
_OFFICIAL_SUFFIXES = ("canada.ca", "europa.eu", "admin.ch", "government.nl")


def is_official_domain(url: str) -> bool:
    """True when the URL's host looks like a government or official body."""

    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    if not host:
        return False
    if host.endswith((".gov", ".mil")) or host in ("gov", "mil"):
        return True
    if _GOV_HOST.search(host):
        return True
    return any(host == d or host.endswith("." + d) for d in _OFFICIAL_SUFFIXES)


def rank_results(agent_id: str, results: list[SearchResult]) -> list[SearchResult]:
    """Visa research puts official sources first (stable); other agents keep their order."""

    if agent_id != "visa":
        return results
    return sorted(results, key=lambda result: not is_official_domain(result.url))


def build_search_client(settings: Settings) -> SearchClient:
    if not settings.search_enabled or settings.search_provider in {"none", "off", "disabled"}:
        return NoopSearchClient()
    if settings.provider == "demo":
        # Demo mode is documented as a no-network smoke test.
        return NoopSearchClient()
    if settings.search_provider == "duckduckgo":
        return DuckDuckGoSearchClient(retries=settings.search_retries)
    return NoopSearchClient()


def queries_for_agent(
    agent_id: str, preferences: TravelPreferences, destination: Destination
) -> list[str]:
    destination_name = destination.display_name
    nationality = preferences.nationality_or_default
    if agent_id == "flight":
        return [
            f"{preferences.departure_city} to {destination_name} flights airlines duration fares",
            f"Google Flights {preferences.departure_city} to {destination_name} best time to book",
            f"Skyscanner {preferences.departure_city} to {destination_name} route airlines",
        ]
    if agent_id == "visa":
        return [
            f"{nationality} passport visa requirements {destination.country} official immigration",
            f"{nationality} evisa visa on arrival {destination.country} official government",
            f"{nationality} travel advice entry requirements {destination.country}",
        ]
    if agent_id == "hotel":
        return [
            f"best neighbourhoods to stay in {destination_name} first time visitor",
            f"Booking.com {destination_name} hotels city centre",
            f"{destination_name} hotel areas near attractions",
        ]
    if agent_id == "experience":
        return [
            f"GetYourGuide {destination_name} top tours",
            f"best things to do in {destination_name} {preferences.days} days",
            f"{destination_name} {preferences.vibe} travel experiences tours",
        ]
    return []


_CITE_RULE = (
    "Cite a source by copying its URL exactly into source_url. If no source supports a claim, "
    'set basis to "model" and source_url to null.'
)


# Most sources shown to one research agent.
SOURCE_LIMIT = 12


@dataclass(frozen=True)
class ShownSource:
    """One source as the research prompt shows it."""

    url: str
    text: str


def shown_sources(
    results: list[SearchResult], pages: Sequence[Page] = (), limit: int = SOURCE_LIMIT
) -> list[ShownSource]:
    """The sources a research prompt shows: read pages first, then snippets, up to `limit`.

    These are also the only URLs a finding may cite, so the two cannot drift apart.
    """

    page_urls = {page.url for page in pages}
    sources = [
        ShownSource(page.url, f"{page.title}\nURL: {page.url}\n{page.text}") for page in pages
    ]
    sources += [
        ShownSource(result.url, f"{result.title}\nURL: {result.url}\nSnippet: {result.snippet}")
        for result in results
        if result.url not in page_urls
    ]
    return sources[:limit]


def format_sources(sources: Sequence[ShownSource]) -> str:
    """The research prompt's source block for sources from `shown_sources`."""

    if not sources:
        return (
            "No live web snippets were available. Be explicit when a claim is based on general "
            "model knowledge rather than a source."
        )
    blocks = ["Use these sources as context."]
    for index, source in enumerate(sources, start=1):
        blocks.append(f"Source [{index}]: {source.text}")
    blocks.append(_CITE_RULE)
    return "\n\n".join(blocks)


def format_search_context(
    results: list[SearchResult], pages: Sequence[Page] = (), limit: int = SOURCE_LIMIT
) -> str:
    return format_sources(shown_sources(results, pages, limit))

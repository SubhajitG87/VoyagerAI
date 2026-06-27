"""Optional public web search snippets for the specialist agents."""

from __future__ import annotations

import logging
from typing import Iterable, List

from voyager_ai.config import Settings
from voyager_ai.models import Destination, SearchResult, TravelPreferences

logger = logging.getLogger(__name__)


class SearchClient:
    def search(self, query: str, max_results: int = 4) -> List[SearchResult]:
        raise NotImplementedError

    def search_many(self, queries: Iterable[str], max_results: int = 4) -> List[SearchResult]:
        seen = set()
        collected = []
        for query in queries:
            for result in self.search(query, max_results=max_results):
                key = result.url.lower()
                if key in seen:
                    continue
                seen.add(key)
                collected.append(result)
        return collected


class NoopSearchClient(SearchClient):
    def search(self, query: str, max_results: int = 4) -> List[SearchResult]:
        return []


class DuckDuckGoSearchClient(SearchClient):
    """Tiny wrapper around the optional ddgs / duckduckgo-search package."""

    def search(self, query: str, max_results: int = 4) -> List[SearchResult]:
        try:
            try:
                from ddgs import DDGS
            except Exception:
                from duckduckgo_search import DDGS
        except Exception:
            logger.warning(
                "The ddgs/duckduckgo-search package is not installed. "
                "Install it with `pip install ddgs` to enable live web snippets. "
                "Falling back to empty search results."
            )
            return []

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
        except Exception:
            return []

        results = []
        for item in raw_results:
            title = str(item.get("title") or item.get("heading") or "").strip()
            url = str(item.get("href") or item.get("url") or "").strip()
            snippet = str(item.get("body") or item.get("snippet") or "").strip()
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results


def build_search_client(settings: Settings) -> SearchClient:
    if not settings.search_enabled or settings.search_provider in {"none", "off", "disabled"}:
        return NoopSearchClient()
    if settings.search_provider == "duckduckgo":
        return DuckDuckGoSearchClient()
    return NoopSearchClient()


def queries_for_agent(
    agent_id: str, preferences: TravelPreferences, destination: Destination
) -> List[str]:
    destination_name = destination.display_name
    nationality = preferences.nationality_or_default
    if agent_id == "flight":
        return [
            "%s to %s flights airlines duration fares" % (preferences.departure_city, destination_name),
            "Google Flights %s to %s best time to book" % (preferences.departure_city, destination_name),
            "Skyscanner %s to %s route airlines" % (preferences.departure_city, destination_name),
        ]
    if agent_id == "visa":
        return [
            "%s passport visa requirements %s official immigration" % (nationality, destination.country),
            "%s evisa visa on arrival %s official government" % (nationality, destination.country),
            "%s travel advice entry requirements %s" % (nationality, destination.country),
        ]
    if agent_id == "hotel":
        return [
            "best neighbourhoods to stay in %s first time visitor" % destination_name,
            "Booking.com %s hotels city centre" % destination_name,
            "%s hotel areas near attractions" % destination_name,
        ]
    if agent_id == "experience":
        return [
            "GetYourGuide %s top tours" % destination_name,
            "best things to do in %s %s days" % (destination_name, preferences.days),
            "%s %s travel experiences tours" % (destination_name, preferences.vibe),
        ]
    return []


def format_search_context(results: List[SearchResult], limit: int = 12) -> str:
    if not results:
        return (
            "No live web snippets were available. Be explicit when a claim is based on general "
            "model knowledge rather than a source."
        )
    lines = ["Use these search snippets as context; cite the URLs when they support a claim:"]
    for result in results[:limit]:
        lines.append(result.to_markdown())
    return "\n".join(lines)

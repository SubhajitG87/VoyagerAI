"""Optional public web search snippets for the specialist agents."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterable

from voyager_ai.config import Settings
from voyager_ai.models import Destination, SearchResult, TravelPreferences

logger = logging.getLogger(__name__)


class SearchClient:
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        raise NotImplementedError

    def search_many(self, queries: Iterable[str], max_results: int = 4) -> list[SearchResult]:
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
    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        return []


class DuckDuckGoSearchClient(SearchClient):
    """Tiny wrapper around the optional ddgs / duckduckgo-search package."""

    def search(self, query: str, max_results: int = 4) -> list[SearchResult]:
        ddgs_module = None
        for module_name in ("ddgs", "duckduckgo_search"):
            if module_name in sys.modules:
                try:
                    ddgs_module = importlib.import_module(module_name)
                    break
                except Exception:
                    continue

        if ddgs_module is None:
            logger.warning(
                "The ddgs/duckduckgo-search package is not installed. "
                "Install it with `pip install ddgs` to enable live web snippets. "
                "Falling back to empty search results."
            )
            return []

        try:
            with ddgs_module.DDGS() as ddgs:
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


def format_search_context(results: list[SearchResult], limit: int = 12) -> str:
    if not results:
        return (
            "No live web snippets were available. Be explicit when a claim is based on general "
            "model knowledge rather than a source."
        )
    lines = ["Use these search snippets as context; cite the URLs when they support a claim:"]
    for result in results[:limit]:
        lines.append(result.to_markdown())
    return "\n".join(lines)

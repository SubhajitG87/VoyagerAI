"""Orchestration for VoyagerAI's multi-agent travel workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeVar

from voyager_ai.config import Settings, load_settings
from voyager_ai.constants import AGENT_SEQUENCE
from voyager_ai.fetch import NoopFetcher, PageFetcher
from voyager_ai.llm import BaseLLMClient, ChunkSink, build_llm_client
from voyager_ai.models import (
    AgentResult,
    Destination,
    TravelPreferences,
    TripPlan,
    ValidationError,
    VoyagerError,
    dedupe_destinations,
)
from voyager_ai.parser import extract_json
from voyager_ai.prompts import (
    DESTINATION_JSON_SCHEMA,
    RESEARCH_CONTEXT_CHARS,
    AgentPrompt,
    discovery_prompt,
    itinerary_max_tokens,
    itinerary_prompt,
    itinerary_tokens_for_context,
    research_chars_for_context,
    research_prompt,
)
from voyager_ai.schemas import ITINERARY_SCHEMA, RESEARCH_SCHEMA
from voyager_ai.search import (
    SearchClient,
    build_search_client,
    format_search_context,
    queries_for_agent,
    rank_results,
)
from voyager_ai.structured import Itinerary, ResearchReport

ProgressCallback = Callable[[str, str, AgentResult | None], None]
T = TypeVar("T")

NO_PAGES_WARNING = "No pages could be read; this agent used search snippets only."
REPAIR_NOTE = (
    "\n\nYour previous reply could not be used: {error}. "
    "Reply again with only JSON that matches the schema."
)


class TravelPlanner:
    """Coordinates discovery, search context, specialist agents, and final itinerary."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: BaseLLMClient | None = None,
        search_client: SearchClient | None = None,
        fetcher: PageFetcher | NoopFetcher | None = None,
    ):
        self.settings = settings or load_settings()
        self.llm = llm_client or build_llm_client(self.settings)
        self.search = search_client or build_search_client(self.settings)
        self.fetcher = fetcher if fetcher is not None else _default_fetcher(self.settings)

    def discover_destinations(
        self, preferences: TravelPreferences, avoid: Iterable[str] = ()
    ) -> list[Destination]:
        prompt = discovery_prompt(preferences, avoid=", ".join(avoid))
        response = self.llm.complete(
            agent_id="discovery",
            model=self.settings.model_for("discovery"),
            system=prompt.system,
            user=prompt.user,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
            json_schema=DESTINATION_JSON_SCHEMA,
        )
        payload = extract_json(response.text)
        items = payload.get("destinations") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise ValidationError("Destination discovery must return a JSON list.")

        destinations = []
        skipped = 0
        for item in items:
            try:
                if isinstance(item, dict):
                    destinations.append(Destination.from_mapping(item))
                else:
                    skipped += 1
            except ValidationError:
                skipped += 1

        destinations = dedupe_destinations(destinations)[:8]
        if not destinations:
            raise VoyagerError(
                "The discovery agent did not return usable destinations. "
                "Try a more common departure city or switch to a larger model."
            )
        if skipped and len(destinations) < 4:
            raise VoyagerError(
                "Most destination suggestions were malformed. Try again or use a stronger "
                "discovery model."
            )
        return destinations

    def research(
        self,
        agent_id: str,
        preferences: TravelPreferences,
        destination: Destination,
        on_chunk: ChunkSink | None = None,
    ) -> AgentResult:
        """Search, read the top pages, then ask one research agent for a structured report."""

        queries = queries_for_agent(agent_id, preferences, destination)
        outcome = self.search.search_many(queries, max_results=self.settings.search_max_results)
        results = rank_results(agent_id, outcome.results)
        warnings = list(outcome.errors)

        pages = []
        for result in results[: self.settings.fetch_pages]:
            page = self.fetcher.fetch(result.url, result.title)
            if page is not None:
                pages.append(page)
        if results and self.settings.fetch_pages > 0 and not pages:
            warnings.append(NO_PAGES_WARNING)

        prompt = research_prompt(
            agent_id, preferences, destination, format_search_context(results, pages)
        )
        allowed_urls = [result.url for result in results]
        report, raw_text = self._complete_structured(
            agent_id,
            prompt,
            RESEARCH_SCHEMA,
            lambda payload: ResearchReport.from_payload(payload, allowed_urls),
            on_chunk,
        )
        return AgentResult(
            agent_id=agent_id,
            model=self.settings.model_for(agent_id),
            report=report,
            warnings=warnings,
            sources=results,
            raw_text=raw_text,
        )

    def build_itinerary(
        self,
        preferences: TravelPreferences,
        destination: Destination,
        research: Mapping[str, AgentResult],
        on_chunk: ChunkSink | None = None,
    ) -> AgentResult:
        """Ask the itinerary agent for a structured plan built from the research reports."""

        itinerary_tokens = self._itinerary_tokens(preferences.days)
        prompt = itinerary_prompt(
            preferences,
            destination,
            {agent_id: result.report for agent_id, result in research.items()},
            research_chars=self._research_chars(itinerary_tokens),
            max_tokens=itinerary_tokens,
        )
        itinerary, raw_text = self._complete_structured(
            "itinerary",
            prompt,
            ITINERARY_SCHEMA,
            lambda payload: Itinerary.from_payload(payload, preferences.days),
            on_chunk,
        )
        return AgentResult(
            agent_id="itinerary",
            model=self.settings.model_for("itinerary"),
            itinerary=itinerary,
            raw_text=raw_text,
        )

    def _complete_structured(
        self,
        agent_id: str,
        prompt: AgentPrompt,
        schema: dict[str, Any],
        parse: Callable[[Any], T],
        on_chunk: ChunkSink | None,
    ) -> tuple[T, str]:
        """Call the model for JSON and parse it, with one repair attempt on a bad reply."""

        def attempt(user: str) -> tuple[T, str]:
            response = self.llm.complete(
                agent_id=agent_id,
                model=self.settings.model_for(agent_id),
                system=prompt.system,
                user=user,
                temperature=prompt.temperature,
                max_tokens=prompt.max_tokens,
                json_schema=schema,
                on_chunk=on_chunk,
            )
            return parse(extract_json(response.text)), response.text

        try:
            return attempt(prompt.user)
        except ValidationError as exc:
            return attempt(prompt.user + REPAIR_NOTE.format(error=exc))

    def _itinerary_tokens(self, days: int) -> int:
        """Reply budget for the itinerary agent, sized to the trip and, on Ollama, the window."""

        if self.settings.provider == "ollama":
            return itinerary_tokens_for_context(days, self.settings.ollama_num_ctx)
        return itinerary_max_tokens(days)

    def _research_chars(self, itinerary_tokens: int) -> int:
        """How much of each research report the itinerary agent is given."""

        if self.settings.research_context_chars > 0:
            return self.settings.research_context_chars
        if self.settings.provider == "ollama":
            return research_chars_for_context(self.settings.ollama_num_ctx, itinerary_tokens)
        return RESEARCH_CONTEXT_CHARS

    def plan_trip(
        self,
        preferences: TravelPreferences,
        destination: Destination,
        progress: ProgressCallback | None = None,
    ) -> TripPlan:
        results: dict[str, AgentResult] = {}

        for agent_id in AGENT_SEQUENCE:
            if progress:
                progress(agent_id, "running", None)
            try:
                if agent_id == "itinerary":
                    result = self.build_itinerary(preferences, destination, results)
                else:
                    result = self.research(agent_id, preferences, destination)
            except Exception as exc:
                result = AgentResult(
                    agent_id=agent_id,
                    model=self.settings.model_for(agent_id),
                    status="error",
                    error=str(exc),
                )
            results[agent_id] = result
            if progress:
                progress(agent_id, result.status, result)

        return TripPlan(preferences=preferences, destination=destination, results=results)


def _default_fetcher(settings: Settings) -> PageFetcher | NoopFetcher:
    """Read pages only when there can be live search results to read."""

    if settings.provider == "demo" or not settings.search_enabled or settings.fetch_pages == 0:
        return NoopFetcher()
    return PageFetcher(max_chars=settings.page_chars)

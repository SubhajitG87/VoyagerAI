"""Orchestration for VoyagerAI's multi-agent travel workflow."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from voyager_ai.config import Settings, load_settings
from voyager_ai.constants import AGENT_SEQUENCE
from voyager_ai.llm import BaseLLMClient, build_llm_client
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
    agent_prompt,
    discovery_prompt,
    itinerary_max_tokens,
    itinerary_tokens_for_context,
    research_chars_for_context,
)
from voyager_ai.search import (
    SearchClient,
    build_search_client,
    format_search_context,
    queries_for_agent,
)

ProgressCallback = Callable[[str, str, AgentResult | None], None]


class TravelPlanner:
    """Coordinates discovery, search context, specialist agents, and final itinerary."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: BaseLLMClient | None = None,
        search_client: SearchClient | None = None,
    ):
        self.settings = settings or load_settings()
        self.llm = llm_client or build_llm_client(self.settings)
        self.search = search_client or build_search_client(self.settings)

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

    def run_agent(
        self,
        agent_id: str,
        preferences: TravelPreferences,
        destination: Destination,
        prior_results: dict[str, str] | None = None,
    ) -> AgentResult:
        prior_results = prior_results or {}
        sources = []
        if agent_id != "itinerary":
            queries = queries_for_agent(agent_id, preferences, destination)
            sources = self.search.search_many(queries, max_results=self.settings.search_max_results)
        search_context = format_search_context(sources)
        itinerary_tokens = self._itinerary_tokens(preferences.days)
        prompt = agent_prompt(
            agent_id,
            preferences,
            destination,
            search_context,
            prior_results,
            research_chars=self._research_chars(itinerary_tokens),
            itinerary_tokens=itinerary_tokens,
        )
        response = self.llm.complete(
            agent_id=agent_id,
            model=self.settings.model_for(agent_id),
            system=prompt.system,
            user=prompt.user,
            temperature=prompt.temperature,
            max_tokens=prompt.max_tokens,
        )
        return AgentResult(
            agent_id=agent_id,
            content=response.text,
            model=response.model,
            status="done",
            sources=sources,
        )

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
        prior_text: dict[str, str] = {}

        for agent_id in AGENT_SEQUENCE:
            if progress:
                progress(agent_id, "running", None)
            try:
                result = self.run_agent(agent_id, preferences, destination, prior_text)
            except Exception as exc:
                result = AgentResult(
                    agent_id=agent_id,
                    content=_fallback_agent_content(agent_id, exc),
                    model=self.settings.model_for(agent_id),
                    status="error",
                    error=str(exc),
                    sources=[],
                )
            results[agent_id] = result
            prior_text[agent_id] = result.content
            if progress:
                progress(agent_id, result.status, result)

        return TripPlan(preferences=preferences, destination=destination, results=results)


def _fallback_agent_content(agent_id: str, exc: Exception) -> str:
    if agent_id == "itinerary":
        return (
            "## Itinerary unavailable\n"
            "The itinerary agent could not complete this run.\n\n"
            f"Reason: {exc}\n\n"
            "You can retry after checking model availability and network/search settings."
        )
    return (
        "## Research unavailable\n"
        "This agent could not complete its research.\n\n"
        f"Reason: {exc}\n\n"
        "The final itinerary agent will continue with the remaining available context."
    )

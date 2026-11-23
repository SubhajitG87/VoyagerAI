"""Orchestration for VoyagerAI's multi-agent travel workflow."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Any, TypeVar

from voyager_ai.config import Settings, load_settings
from voyager_ai.constants import AGENT_SEQUENCE
from voyager_ai.events import EventSink, PlanEvent
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

T = TypeVar("T")

RESEARCH_AGENTS = tuple(agent_id for agent_id in AGENT_SEQUENCE if agent_id != "itinerary")
UNKNOWN_ERROR = "unknown error"
# How long the drain loop waits for an event before checking whether the agents are done.
DRAIN_POLL_S = 0.05

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
        # Every LLM call takes a slot, so a local server is not asked for more than it can run.
        self._llm_slots = threading.Semaphore(self.settings.effective_llm_concurrency)

    def discover_destinations(
        self, preferences: TravelPreferences, avoid: Iterable[str] = ()
    ) -> list[Destination]:
        prompt = discovery_prompt(preferences, avoid=", ".join(avoid))
        with self._llm_slots:
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
        fetching = self.settings.fetch_pages > 0 and not isinstance(self.fetcher, NoopFetcher)
        if results and fetching and not pages:
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
        """Call the model for JSON and parse it, with one repair attempt on a bad reply.

        The repair attempt is not streamed, so a consumer never sees two replies as one stream.
        """

        def attempt(user: str, sink: ChunkSink | None) -> tuple[T, str]:
            with self._llm_slots:
                response = self.llm.complete(
                    agent_id=agent_id,
                    model=self.settings.model_for(agent_id),
                    system=prompt.system,
                    user=user,
                    temperature=prompt.temperature,
                    max_tokens=prompt.max_tokens,
                    json_schema=schema,
                    on_chunk=sink,
                )
            return parse(extract_json(response.text)), response.text

        try:
            return attempt(prompt.user, on_chunk)
        except ValidationError as exc:
            return attempt(prompt.user + REPAIR_NOTE.format(error=exc), None)

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
        on_event: EventSink | None = None,
    ) -> TripPlan:
        """Run the four research agents in parallel, then the itinerary agent.

        Events are queued by the worker threads and passed to `on_event` on the calling
        thread only. `plan_finished` is always the last event. An `Exception` in an agent
        becomes that agent's error; any other `BaseException` raised in a worker (such as a
        test's network guard) also fails the agent, and is re-raised after `plan_finished`.
        """

        emit: EventSink = on_event or (lambda event: None)
        events: queue.Queue[PlanEvent] = queue.Queue()
        results: dict[str, AgentResult] = {}
        fatal: list[BaseException] = []
        executor = ThreadPoolExecutor(
            max_workers=len(RESEARCH_AGENTS), thread_name_prefix="voyager-agent"
        )
        try:
            research = {
                agent_id: executor.submit(
                    self._run_agent,
                    agent_id,
                    partial(self.research, agent_id, preferences, destination),
                    events,
                    fatal,
                )
                for agent_id in RESEARCH_AGENTS
            }
            _drain(events, list(research.values()), emit)
            for agent_id, future in research.items():
                results[agent_id] = future.result()

            itinerary = executor.submit(
                self._run_agent,
                "itinerary",
                partial(self.build_itinerary, preferences, destination, dict(results)),
                events,
                fatal,
            )
            _drain(events, [itinerary], emit)
            results["itinerary"] = itinerary.result()
        except BaseException:
            # Interrupted, or on_event raised: drop queued work and do not wait for
            # calls already running; their events go nowhere.
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown()

        plan = TripPlan(preferences=preferences, destination=destination, results=results)
        emit(PlanEvent("plan_finished", plan=plan))
        if fatal:
            raise fatal[0]
        return plan

    def _run_agent(
        self,
        agent_id: str,
        run: Callable[[ChunkSink], AgentResult],
        events: queue.Queue[PlanEvent],
        fatal: list[BaseException],
    ) -> AgentResult:
        """Worker body: run one agent, queueing its events. Never raises.

        A `BaseException` that is not an `Exception` is appended to `fatal` for the caller
        to re-raise once the plan is finished.
        """

        events.put(PlanEvent("agent_started", agent_id))
        try:
            result = run(lambda text: events.put(PlanEvent("agent_chunk", agent_id, text=text)))
        except BaseException as exc:
            if not isinstance(exc, Exception):
                fatal.append(exc)
            result = AgentResult(
                agent_id=agent_id,
                model=self.settings.model_for(agent_id),
                status="error",
                error=str(exc) or type(exc).__name__,
            )
        if result.status == "error":
            text = result.error or UNKNOWN_ERROR
            events.put(PlanEvent("agent_failed", agent_id, text=text, result=result))
        else:
            events.put(PlanEvent("agent_finished", agent_id, result=result))
        return result


def _drain(
    events: queue.Queue[PlanEvent], futures: list[Future[AgentResult]], emit: EventSink
) -> None:
    """Pass queued events to `emit` until every future is done and the queue is empty.

    A worker queues all of its events before its future is done, so once every future is
    done, whatever is left in the queue is the last of it.
    """

    while True:
        try:
            event = events.get(timeout=DRAIN_POLL_S)
        except queue.Empty:
            if all(future.done() for future in futures):
                break
            continue
        emit(event)
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            return
        emit(event)


def _default_fetcher(settings: Settings) -> PageFetcher | NoopFetcher:
    """Read pages only when there can be live search results to read."""

    if settings.provider == "demo" or not settings.search_enabled or settings.fetch_pages == 0:
        return NoopFetcher()
    return PageFetcher(max_chars=settings.page_chars)

import json
import threading
import time

import pytest

from voyager_ai import planner as planner_module
from voyager_ai.config import Settings
from voyager_ai.demo import demo_itinerary_payload
from voyager_ai.events import PlanEvent
from voyager_ai.fetch import NoopFetcher
from voyager_ai.llm import DemoLLMClient, LLMError, LLMResponse
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.planner import TravelPlanner
from voyager_ai.search import NoopSearchClient

DEST = Destination("Tokyo", "Japan", "Chosen by test.")
PREFS = TravelPreferences("Mumbai", nationality="Indian")
RESEARCH = ("flight", "visa", "hotel", "experience")


def _demo_planner(llm=None, provider="demo"):
    return TravelPlanner(
        Settings(provider=provider),
        llm_client=llm or DemoLLMClient(),
        search_client=NoopSearchClient(),
        fetcher=NoopFetcher(),
    )


def _plan_within(planner, seconds=10.0, on_event=None):
    """Run plan_trip on a helper thread so a deadlock fails the test instead of hanging it."""

    outcome = {}

    def run():
        try:
            outcome["trip"] = planner.plan_trip(PREFS, DEST, on_event=on_event)
        except BaseException as exc:  # surfaced to the test below
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "plan_trip did not finish in time"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["trip"]


class BarrierLLM(DemoLLMClient):
    """Research calls only return once all four are in flight at the same time."""

    def __init__(self):
        self.barrier = threading.Barrier(4, timeout=2)

    def complete(self, *, agent_id, **kwargs):
        if agent_id != "itinerary":
            self.barrier.wait()
        return super().complete(agent_id=agent_id, **kwargs)


class InFlightLLM(DemoLLMClient):
    """Records the most LLM calls seen at once; the flight agent's first reply needs repair."""

    def __init__(self):
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.flight_calls = 0

    def complete(self, *, agent_id, **kwargs):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if agent_id == "flight":
                self.flight_calls += 1
                broken = self.flight_calls == 1
            else:
                broken = False
        try:
            time.sleep(0.05)
            response = super().complete(agent_id=agent_id, **kwargs)
            if broken:
                response.text = "not json"
            return response
        finally:
            with self.lock:
                self.in_flight -= 1


class FailingLLM(DemoLLMClient):
    """One agent streams a chunk and then fails with the given exception."""

    def __init__(self, agent_id, exc):
        self.failing = agent_id
        self.exc = exc

    def complete(self, *, agent_id, on_chunk=None, **kwargs):
        if agent_id == self.failing:
            if on_chunk is not None:
                on_chunk("partial")
            raise self.exc
        return super().complete(agent_id=agent_id, on_chunk=on_chunk, **kwargs)


class StreamingScriptLLM(DemoLLMClient):
    """Streams scripted itinerary replies and records whether each call was streamed."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.streamed = []

    def complete(self, *, agent_id, on_chunk=None, **kwargs):
        if agent_id != "itinerary":
            return super().complete(agent_id=agent_id, on_chunk=on_chunk, **kwargs)
        self.streamed.append(on_chunk is not None)
        text = self.replies.pop(0)
        if on_chunk is not None:
            on_chunk(text)
        return LLMResponse(text=text, model="demo", raw={})


class Interrupted(BaseException):
    pass


def test_event_order_and_thread():
    events, threads = [], set()

    def sink(e):
        events.append(e)
        threads.add(threading.get_ident())

    _demo_planner().plan_trip(PREFS, DEST, on_event=sink)
    assert threads == {threading.get_ident()}
    assert [e.kind for e in events].count("plan_finished") == 1
    assert events[-1].kind == "plan_finished" and events[-1].plan is not None
    for agent in (*RESEARCH, "itinerary"):
        kinds = [e.kind for e in events if e.agent_id == agent]
        assert kinds[0] == "agent_started" and kinds[-1] == "agent_finished"
        assert set(kinds[1:-1]) <= {"agent_chunk"}
        assert "agent_chunk" in kinds


def test_itinerary_starts_after_all_research_finishes():
    events = []
    _demo_planner().plan_trip(PREFS, DEST, on_event=events.append)
    kinds = [(e.kind, e.agent_id) for e in events]
    itinerary_start = kinds.index(("agent_started", "itinerary"))
    for agent in RESEARCH:
        assert kinds.index(("agent_finished", agent)) < itinerary_start


def test_research_agents_run_concurrently_on_hosted_providers():
    trip = _plan_within(_demo_planner(BarrierLLM(), provider="openai-compatible"))
    assert trip.errors == {}
    assert list(trip.results) == [*RESEARCH, "itinerary"]


def test_ollama_runs_one_llm_call_at_a_time():
    llm = InFlightLLM()
    trip = _plan_within(_demo_planner(llm, provider="ollama"))
    assert trip.errors == {} and llm.flight_calls == 2
    assert llm.max_in_flight == 1


def test_configured_concurrency_limits_llm_calls():
    llm = InFlightLLM()
    planner = TravelPlanner(
        Settings(provider="openai-compatible", llm_concurrency=2),
        llm_client=llm,
        search_client=NoopSearchClient(),
        fetcher=NoopFetcher(),
    )
    assert _plan_within(planner).errors == {}
    assert llm.max_in_flight == 2


def test_failed_agent_emits_failed_after_its_chunks_and_plan_continues():
    events = []
    llm = FailingLLM("visa", LLMError("stream interrupted"))
    trip = _demo_planner(llm).plan_trip(PREFS, DEST, on_event=events.append)
    kinds = [e.kind for e in events if e.agent_id == "visa"]
    assert kinds == ["agent_started", "agent_chunk", "agent_failed"]
    failed = [e for e in events if e.kind == "agent_failed"][0]
    assert "stream interrupted" in failed.text and failed.result.status == "error"
    assert trip.results["itinerary"].status == "done"
    assert events[-1].kind == "plan_finished"


def test_failed_llm_call_releases_its_slot():
    # One slot on ollama: a failing call that kept its slot would block every other agent.
    llm = FailingLLM("flight", LLMError("down"))
    trip = _plan_within(_demo_planner(llm, provider="ollama"))
    assert set(trip.errors) == {"flight"}


def test_unexpected_worker_exception_becomes_agent_failed():
    events = []
    llm = FailingLLM("hotel", Interrupted("worker blew up"))
    trip = _plan_within(_demo_planner(llm), on_event=events.append)
    assert trip.results["hotel"].status == "error"
    assert [e.kind for e in events if e.agent_id == "hotel"][-1] == "agent_failed"
    assert [e.kind for e in events].count("plan_finished") == 1
    assert events[-1].kind == "plan_finished"


def test_repair_attempt_is_not_streamed():
    valid = json.dumps(demo_itinerary_payload(PREFS.days))
    llm = StreamingScriptLLM(["not json", valid])
    events = []
    trip = _demo_planner(llm).plan_trip(PREFS, DEST, on_event=events.append)
    assert trip.itinerary is not None and llm.streamed == [True, False]
    chunks = [e.text for e in events if e.kind == "agent_chunk" and e.agent_id == "itinerary"]
    assert "".join(chunks) == "not json"


class RecordingExecutor(planner_module.ThreadPoolExecutor):
    shutdowns: list = []

    def shutdown(self, wait=True, *, cancel_futures=False):
        RecordingExecutor.shutdowns.append(cancel_futures)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("sink broke")])
def test_caller_exception_cancels_and_propagates(monkeypatch, exc):
    RecordingExecutor.shutdowns = []
    monkeypatch.setattr(planner_module, "ThreadPoolExecutor", RecordingExecutor)

    def sink(event):
        raise exc

    with pytest.raises(type(exc)):
        _demo_planner().plan_trip(PREFS, DEST, on_event=sink)
    assert RecordingExecutor.shutdowns == [True]


def test_executor_is_shut_down_after_a_normal_plan(monkeypatch):
    RecordingExecutor.shutdowns = []
    monkeypatch.setattr(planner_module, "ThreadPoolExecutor", RecordingExecutor)
    _demo_planner().plan_trip(PREFS, DEST)
    assert RecordingExecutor.shutdowns == [False]


def test_events_serialise_to_json():
    _demo_planner().plan_trip(PREFS, DEST, on_event=lambda e: json.dumps(e.to_dict()))


def test_finished_event_dict_has_result_summary():
    events = []
    trip = _demo_planner().plan_trip(PREFS, DEST, on_event=events.append)
    visa = next(e for e in events if e.kind == "agent_finished" and e.agent_id == "visa")
    data = visa.to_dict()
    report = trip.results["visa"].report
    assert data["kind"] == "agent_finished" and data["agent_id"] == "visa"
    assert data["result"] == {
        "status": "done",
        "model": trip.results["visa"].model,
        "error": None,
        "warnings": [],
        "findings": len(report.findings),
        "sourced": report.sourced_count,
    }
    assert "plan" not in events[-1].to_dict()
    assert PlanEvent("agent_started", "flight").to_dict()["result"] is None

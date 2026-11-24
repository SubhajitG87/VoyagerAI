import json

import pytest

from voyager_ai.config import Settings
from voyager_ai.demo import demo_itinerary_payload
from voyager_ai.fetch import NoopFetcher
from voyager_ai.llm import DemoLLMClient, LLMError
from voyager_ai.models import Destination, SearchResult, TravelPreferences, ValidationError
from voyager_ai.planner import TravelPlanner
from voyager_ai.prompts import BREVITY_NOTE
from voyager_ai.render import render_markdown
from voyager_ai.search import NoopSearchClient, SearchClient, SearchError

DEST = Destination("Tokyo", "Japan", "Chosen by test.")
PREFS = TravelPreferences("Mumbai", nationality="Indian")


class ScriptedLLM(DemoLLMClient):
    """Replies from a per-agent script, falling back to the demo client."""

    def __init__(self, script=None):
        self.script = {agent_id: list(replies) for agent_id, replies in (script or {}).items()}
        self.calls = {}
        self.last_user = {}

    def complete(self, *, agent_id, user, **kwargs):
        self.calls[agent_id] = self.calls.get(agent_id, 0) + 1
        self.last_user[agent_id] = user
        response = super().complete(agent_id=agent_id, user=user, **kwargs)
        replies = self.script.get(agent_id)
        if replies:
            response.text = replies.pop(0)
        return response


class VisaOnlySearch(SearchClient):
    def search(self, query, max_results=4):
        if "visa" in query.lower():
            return [SearchResult("UK visa page", "https://www.gov.uk/x", "Entry rules.")]
        return []


class TwoResultSearch(SearchClient):
    def search(self, query, max_results=4):
        return [
            SearchResult("A", "https://a.example/1", "a"),
            SearchResult("B", "https://b.example/2", "b"),
        ]


class UnreadableFetcher:
    def __init__(self):
        self.urls = []

    def fetch(self, url, title=""):
        self.urls.append(url)
        return None


def _demo_planner(llm=None, search=None, fetcher=None):
    return TravelPlanner(
        Settings(provider="demo"),
        llm_client=llm or DemoLLMClient(),
        search_client=search or NoopSearchClient(),
        fetcher=fetcher,
    )


def test_demo_plan_is_structured_end_to_end():
    trip = _demo_planner().plan_trip(TravelPreferences("Mumbai", nights=30), DEST)
    assert not trip.errors and len(trip.itinerary.days) == 31
    assert trip.results["visa"].report.findings


def test_invented_source_is_downgraded_end_to_end():
    visa = json.dumps(
        {
            "summary": "Visa summary.",
            "findings": [
                {
                    "topic": "Entry",
                    "claim": "Visa-free for 90 days.",
                    "basis": "sourced",
                    "source_url": "https://www.gov.uk/x",
                },
                {
                    "topic": "Fees",
                    "claim": "No fee.",
                    "basis": "sourced",
                    "source_url": "https://fake.example/y",
                },
            ],
        }
    )
    llm = ScriptedLLM({"visa": [visa]})
    trip = _demo_planner(llm=llm, search=VisaOnlySearch()).plan_trip(PREFS, DEST)
    bases = [f.basis for f in trip.results["visa"].report.findings]
    assert bases == ["sourced", "model"]
    assert [s.url for s in trip.results["visa"].sources] == ["https://www.gov.uk/x"]


def test_invalid_json_gets_one_repair_attempt():
    valid = json.dumps(demo_itinerary_payload(PREFS.days))
    llm = ScriptedLLM({"itinerary": ["not json", valid]})
    trip = _demo_planner(llm=llm).plan_trip(PREFS, DEST)
    assert trip.itinerary is not None and llm.calls["itinerary"] == 2
    assert "could not be used" in llm.last_user["itinerary"]
    assert trip.results["itinerary"].raw_text == valid


def test_itinerary_failure_keeps_research_in_markdown():
    llm = ScriptedLLM({"itinerary": ["not json", "not json"]})
    trip = _demo_planner(llm=llm).plan_trip(PREFS, DEST)
    md = render_markdown(trip)
    assert trip.results["itinerary"].status == "error" and "## Visa Advisor" in md
    assert "> Agent error:" in md
    assert trip.itinerary is None and "itinerary" in trip.errors


def test_unreadable_pages_add_a_warning():
    fetcher = UnreadableFetcher()
    trip = _demo_planner(search=TwoResultSearch(), fetcher=fetcher).plan_trip(PREFS, DEST)
    assert (
        "No pages could be read; this agent used search snippets only."
        in trip.results["flight"].warnings
    )
    assert fetcher.urls[:2] == ["https://a.example/1", "https://b.example/2"]


def test_no_search_results_means_no_unreadable_warning():
    trip = _demo_planner().plan_trip(PREFS, DEST)
    assert trip.results["flight"].warnings == []


def test_fetched_pages_reach_the_research_prompt():
    from voyager_ai.fetch import Page

    class OnePageFetcher:
        def fetch(self, url, title=""):
            return Page(url, title, "PAGE-TEXT-MARKER") if url.endswith("/1") else None

    llm = ScriptedLLM()
    settings = Settings(provider="demo", fetch_pages=1)
    planner = TravelPlanner(
        settings, llm_client=llm, search_client=TwoResultSearch(), fetcher=OnePageFetcher()
    )
    trip = planner.plan_trip(PREFS, DEST)
    assert "PAGE-TEXT-MARKER" in llm.last_user["flight"]
    assert trip.results["flight"].warnings == []


def test_default_fetcher_is_noop_unless_fetching_can_happen():
    from voyager_ai.fetch import NoopFetcher, PageFetcher

    def fetcher_for(**kwargs):
        return TravelPlanner(
            Settings(**kwargs), llm_client=DemoLLMClient(), search_client=NoopSearchClient()
        ).fetcher

    assert isinstance(fetcher_for(provider="demo"), NoopFetcher)
    assert isinstance(fetcher_for(provider="ollama", search_enabled=False), NoopFetcher)
    assert isinstance(fetcher_for(provider="ollama", fetch_pages=0), NoopFetcher)
    real = fetcher_for(provider="ollama", page_chars=900)
    assert isinstance(real, PageFetcher) and real.max_chars == 900


def test_no_unreadable_warning_when_fetching_is_off():
    trip = _demo_planner(search=TwoResultSearch(), fetcher=NoopFetcher()).plan_trip(PREFS, DEST)
    assert trip.results["flight"].warnings == []


class CutOffLLM(DemoLLMClient):
    """Itinerary replies from a script of (text, truncated); records each call's request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def complete(self, *, agent_id, user, max_tokens=2800, **kwargs):
        response = super().complete(agent_id=agent_id, user=user, max_tokens=max_tokens, **kwargs)
        if agent_id == "itinerary":
            self.requests.append({"user": user, "max_tokens": max_tokens})
            response.text, response.truncated = self.replies.pop(0)
        return response


def _build_itinerary(llm, settings=None):
    planner = TravelPlanner(
        settings or Settings(provider="demo"),
        llm_client=llm,
        search_client=NoopSearchClient(),
        fetcher=NoopFetcher(),
    )
    return planner.build_itinerary(PREFS, DEST, {})


def test_cut_off_reply_gets_a_shorter_repair_with_more_tokens():
    valid = json.dumps(demo_itinerary_payload(PREFS.days))
    llm = CutOffLLM([('{"days": [', True), (valid, False)])
    result = _build_itinerary(llm)
    assert result.itinerary is not None
    first, repair = llm.requests
    assert "cut off" in repair["user"] and BREVITY_NOTE in repair["user"]
    assert repair["max_tokens"] == first["max_tokens"] * 3 // 2


def test_reply_that_was_not_cut_off_keeps_the_plain_repair():
    valid = json.dumps(demo_itinerary_payload(PREFS.days))
    llm = CutOffLLM([("not json", False), (valid, False)])
    _build_itinerary(llm)
    first, repair = llm.requests
    assert "could not be used" in repair["user"] and "cut off" not in repair["user"]
    assert repair["max_tokens"] == first["max_tokens"]


def test_ollama_repair_tokens_stay_inside_the_window():
    valid = json.dumps(demo_itinerary_payload(PREFS.days))
    llm = CutOffLLM([('{"days": [', True), (valid, False)])
    _build_itinerary(llm, Settings(provider="ollama", ollama_num_ctx=8192))
    first, repair = llm.requests
    # 4200 * 1.5 would be 6300; the 8192 window leaves 6214 (see itinerary_tokens_for_context).
    assert (first["max_tokens"], repair["max_tokens"]) == (4200, 6214)
    assert repair["max_tokens"] + len(repair["user"]) // 3 <= 8192


def test_repair_that_is_also_cut_off_says_so():
    llm = CutOffLLM([('{"days": [', True), ('{"days": [', True)])
    with pytest.raises(ValidationError, match="the reply was cut off at the token limit"):
        _build_itinerary(llm)


class ManyResultSearch(SearchClient):
    """Thirteen distinct results for the visa agent, one more than the prompt shows."""

    def search(self, query, max_results=4):
        if "visa requirements" not in query:
            return []
        return [SearchResult(f"R{i}", f"https://r.example/{i}", "s") for i in range(1, 14)]


def _visa_citing(*urls):
    findings = [
        {"topic": "T", "claim": f"Claim {url}", "basis": "sourced", "source_url": url}
        for url in urls
    ]
    return json.dumps({"summary": "S.", "findings": findings})


def test_only_sources_shown_in_the_prompt_can_be_cited():
    llm = ScriptedLLM({"visa": [_visa_citing("https://r.example/12", "https://r.example/13")]})
    trip = _demo_planner(llm=llm, search=ManyResultSearch()).plan_trip(PREFS, DEST)
    assert "https://r.example/13" not in llm.last_user["visa"]
    bases = [f.basis for f in trip.results["visa"].report.findings]
    assert bases == ["sourced", "model"]


class FailingVisaLLM(DemoLLMClient):
    def complete(self, *, agent_id, **kwargs):
        if agent_id == "visa":
            raise LLMError("visa model is down")
        return super().complete(agent_id=agent_id, **kwargs)


class HalfBrokenSearch(VisaOnlySearch):
    def search(self, query, max_results=4):
        if "evisa" in query:
            raise SearchError("rate limited")
        return super().search(query, max_results)


def test_failed_agent_keeps_its_search_warnings_and_sources():
    planner = _demo_planner(llm=FailingVisaLLM(), search=HalfBrokenSearch())
    visa = planner.plan_trip(PREFS, DEST).results["visa"]
    assert visa.status == "error" and "visa model is down" in visa.error
    assert [s.url for s in visa.sources] == ["https://www.gov.uk/x"]
    assert any("rate limited" in w for w in visa.warnings)
    assert "> Warning: Search failed" in render_markdown(_trip_with(visa))


def _trip_with(result):
    from voyager_ai.models import TripPlan

    return TripPlan(PREFS, DEST, {"visa": result})

from voyager_ai.config import Settings
from voyager_ai.llm import DemoLLMClient
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.planner import TravelPlanner
from voyager_ai.search import NoopSearchClient


class LongResearchLLM(DemoLLMClient):
    """Returns 9,000 characters of research per agent and records every prompt."""

    def __init__(self):
        self.users = {}
        self.max_tokens = {}

    def complete(self, *, agent_id, user, **kwargs):
        self.users[agent_id] = user
        self.max_tokens[agent_id] = kwargs.get("max_tokens")
        response = super().complete(agent_id=agent_id, user=user, **kwargs)
        if agent_id in {"flight", "visa", "hotel", "experience"}:
            response.text = "R" * 9000
        return response


def _run(settings, nights=7):
    llm = LongResearchLLM()
    planner = TravelPlanner(settings, llm_client=llm, search_client=NoopSearchClient())
    prefs = TravelPreferences("Mumbai", nights=nights)
    trip = planner.plan_trip(prefs, Destination("Tokyo", "Japan", "Test."))
    return llm, trip


def _itinerary_prompt_for(settings):
    return _run(settings)[0].users["itinerary"]


def test_ollama_research_is_trimmed_to_fit_a_small_context():
    prompt = _itinerary_prompt_for(Settings(provider="ollama", ollama_num_ctx=8192))
    assert "R" * 2607 + "..." in prompt
    assert "R" * 2611 not in prompt


def test_hosted_provider_gets_full_research_by_default():
    prompt = _itinerary_prompt_for(Settings(provider="openai-compatible"))
    assert "R" * 9000 in prompt


def test_explicit_research_budget_wins_for_any_provider():
    settings = Settings(provider="openai-compatible", research_context_chars=1500)
    prompt = _itinerary_prompt_for(settings)
    assert "R" * 1497 + "..." in prompt
    assert "R" * 1501 not in prompt


def test_month_long_trip_on_ollama_trades_research_for_reply_room():
    llm, _ = _run(Settings(provider="ollama", ollama_num_ctx=16384), nights=30)
    assert llm.max_tokens["itinerary"] == 9260
    assert "R" * 4956 + "..." in llm.users["itinerary"]
    assert "R" * 4960 not in llm.users["itinerary"]


def test_month_long_trip_on_hosted_provider_gets_full_reply_budget():
    llm, _ = _run(Settings(provider="openai-compatible"), nights=30)
    assert llm.max_tokens["itinerary"] == 9260


def test_demo_itinerary_covers_every_day_of_a_long_trip():
    _, trip = _run(Settings(provider="demo"), nights=30)
    assert "## Day 31 -" in trip.itinerary
    assert "## Budget Breakdown" in trip.itinerary

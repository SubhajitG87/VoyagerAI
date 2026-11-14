from voyager_ai.config import Settings
from voyager_ai.llm import DemoLLMClient
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.planner import TravelPlanner
from voyager_ai.search import NoopSearchClient


class LongResearchLLM(DemoLLMClient):
    """Returns 9,000 characters of research per agent and records every prompt."""

    def __init__(self):
        self.users = {}

    def complete(self, *, agent_id, user, **kwargs):
        self.users[agent_id] = user
        response = super().complete(agent_id=agent_id, user=user, **kwargs)
        if agent_id in {"flight", "visa", "hotel", "experience"}:
            response.text = "R" * 9000
        return response


def _itinerary_prompt_for(settings):
    llm = LongResearchLLM()
    planner = TravelPlanner(settings, llm_client=llm, search_client=NoopSearchClient())
    planner.plan_trip(TravelPreferences("Mumbai"), Destination("Tokyo", "Japan", "Test."))
    return llm.users["itinerary"]


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

import unittest

from voyager_ai.config import Settings
from voyager_ai.llm import DemoLLMClient
from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.planner import TravelPlanner
from voyager_ai.search import NoopSearchClient, SearchClient, SearchError


class PlannerDemoTests(unittest.TestCase):
    def test_demo_discovery_and_plan(self):
        settings = Settings(provider="demo", search_enabled=False)
        planner = TravelPlanner(
            settings,
            llm_client=DemoLLMClient(),
            search_client=NoopSearchClient(),
        )
        prefs = TravelPreferences("Mumbai", nationality="Indian", vibe="food")
        destinations = planner.discover_destinations(prefs)
        self.assertEqual(len(destinations), 8)

        trip = planner.plan_trip(prefs, Destination("Tokyo", "Japan", "Chosen by test."))
        self.assertIn("Day 1", trip.itinerary)
        self.assertFalse(trip.errors)

    def test_search_warnings_reach_the_trip_markdown(self):
        class DownSearch(SearchClient):
            def search(self, query, max_results=4):
                raise SearchError("down")

        planner = TravelPlanner(
            Settings(provider="demo", search_enabled=False),
            llm_client=DemoLLMClient(),
            search_client=DownSearch(),
        )
        prefs = TravelPreferences("Mumbai", nationality="Indian", vibe="food")
        trip = planner.plan_trip(prefs, Destination("Tokyo", "Japan", "Chosen by test."))
        self.assertTrue(trip.results["flight"].warnings)
        self.assertIn("> Search warning:", trip.to_markdown())


if __name__ == "__main__":
    unittest.main()

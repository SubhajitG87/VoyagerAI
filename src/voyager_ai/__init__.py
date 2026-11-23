"""VoyagerAI: multi-agent travel research in Python."""

from voyager_ai.models import Destination, TravelPreferences, TripPlan
from voyager_ai.planner import TravelPlanner

__all__ = ["Destination", "TravelPreferences", "TravelPlan", "TripPlan", "TravelPlanner"]

__version__ = "0.2.0"

# Backwards-compatible alias for users who expect "plan" terminology.
TravelPlan = TripPlan

"""Booking deep links, and the seam where a future fare or rate API plugs in.

Flight and hotel figures in a plan are sourced estimates. These links let the traveller check
live prices themselves; a `PriceProvider` can add real quotes once one is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from urllib.parse import quote

from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.structured import CostRange


def deep_links(preferences: TravelPreferences, destination: Destination) -> dict[str, str]:
    """Search links for this trip on Google Flights, Skyscanner and Booking.com."""

    flights = f"Flights from {preferences.departure_city} to {destination.display_name}"
    return {
        "Google Flights": f"https://www.google.com/travel/flights?q={quote(flights, safe='')}",
        # Skyscanner's deep links need airport codes, which the app does not collect yet.
        "Skyscanner": "https://www.skyscanner.net/",
        "Booking.com": (
            "https://www.booking.com/searchresults.html?ss="
            f"{quote(destination.display_name, safe='')}"
        ),
    }


@dataclass
class PriceQuote:
    label: str
    cost: CostRange
    url: str


class PriceProvider(Protocol):
    def quotes(
        self, preferences: TravelPreferences, destination: Destination
    ) -> list[PriceQuote]: ...


class NoPrices:
    """The default provider: no live prices, so the plan shows booking links only."""

    def quotes(self, preferences: TravelPreferences, destination: Destination) -> list[PriceQuote]:
        return []

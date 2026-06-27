"""Domain models and validation for VoyagerAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from voyager_ai.constants import AGENTS, VIBE_DESCRIPTIONS
from voyager_ai.utils import clean_text, date_range


class VoyagerError(Exception):
    """Base exception for user-facing VoyagerAI failures."""


class ValidationError(VoyagerError, ValueError):
    """Raised when user input cannot produce a safe trip plan."""


@dataclass
class TravelPreferences:
    """Inputs collected in the original JSX form."""

    departure_city: str
    nationality: str = ""
    months: int = 6
    direct_flights: bool = False
    nights: int = 7
    vibe: str = "city"

    def __post_init__(self) -> None:
        self.departure_city = clean_text(self.departure_city)
        self.nationality = clean_text(self.nationality)
        self.months = int(self.months)
        self.nights = int(self.nights)
        self.vibe = clean_text(self.vibe).lower() or "city"

        errors = []
        if not self.departure_city:
            errors.append("Departure city is required.")
        if self.months not in {3, 6, 12}:
            errors.append("Travel window must be 3, 6, or 12 months.")
        if not 3 <= self.nights <= 30:
            errors.append("Trip length must be between 3 and 30 nights.")
        if self.vibe not in VIBE_DESCRIPTIONS:
            errors.append("Travel style must be one of: %s." % ", ".join(VIBE_DESCRIPTIONS))
        if errors:
            raise ValidationError(" ".join(errors))

    @property
    def days(self) -> int:
        return self.nights + 1

    @property
    def travel_window(self) -> str:
        return date_range(self.months)

    @property
    def vibe_description(self) -> str:
        return VIBE_DESCRIPTIONS[self.vibe]

    @property
    def flight_preference(self) -> str:
        return "direct flights only" if self.direct_flights else "direct or connecting flights"

    @property
    def nationality_or_default(self) -> str:
        return self.nationality or "international traveller"


@dataclass
class Destination:
    """A destination suggestion returned by the discovery agent."""

    name: str
    country: str
    why: str
    tags: List[str] = field(default_factory=list)
    flag: str = ""

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "Destination":
        name = clean_text(payload.get("name"))
        country = clean_text(payload.get("country"))
        why = clean_text(payload.get("why"))
        flag = clean_text(payload.get("flag"))
        raw_tags = payload.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = [raw_tags]
        tags = [clean_text(tag) for tag in raw_tags if clean_text(tag)][:3]

        if not name or not country or not why:
            raise ValidationError("Destination suggestions must include name, country, and why.")
        return cls(name=name, country=country, why=why, tags=tags, flag=flag)

    @property
    def display_name(self) -> str:
        return "%s, %s" % (self.name, self.country)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "country": self.country,
            "flag": self.flag,
            "why": self.why,
            "tags": list(self.tags),
        }


@dataclass
class SearchResult:
    """A web search snippet passed into an agent prompt."""

    title: str
    url: str
    snippet: str = ""

    def to_markdown(self) -> str:
        snippet = (" - " + self.snippet) if self.snippet else ""
        return "- %s: %s%s" % (self.title, self.url, snippet)


@dataclass
class AgentResult:
    """Output from one specialist agent."""

    agent_id: str
    content: str
    model: str
    status: str = "done"
    error: Optional[str] = None
    sources: List[SearchResult] = field(default_factory=list)

    @property
    def label(self) -> str:
        return AGENTS.get(self.agent_id, {}).get("label", self.agent_id.title())


@dataclass
class TripPlan:
    """A complete trip plan and the supporting agent research."""

    preferences: TravelPreferences
    destination: Destination
    results: Dict[str, AgentResult]

    @property
    def itinerary(self) -> str:
        result = self.results.get("itinerary")
        return result.content if result else ""

    @property
    def errors(self) -> Dict[str, str]:
        return {
            agent_id: result.error
            for agent_id, result in self.results.items()
            if result.error is not None
        }

    def to_markdown(self) -> str:
        lines = [
            "# VoyagerAI Trip Plan: %s" % self.destination.display_name,
            "",
            "- Departure: %s" % self.preferences.departure_city,
            "- Nationality: %s" % self.preferences.nationality_or_default,
            "- Travel window: %s" % self.preferences.travel_window,
            "- Trip length: %s nights / %s days" % (self.preferences.nights, self.preferences.days),
            "- Flight preference: %s" % self.preferences.flight_preference,
            "- Travel style: %s" % self.preferences.vibe_description,
            "",
        ]

        for agent_id in ("flight", "visa", "hotel", "experience", "itinerary"):
            result = self.results.get(agent_id)
            if not result:
                continue
            lines.extend(["## %s" % result.label, "", result.content.strip(), ""])
            if result.sources:
                lines.extend(["### Sources Considered", ""])
                lines.extend(source.to_markdown() for source in result.sources)
                lines.append("")
            if result.error:
                lines.extend(["> Agent warning: %s" % result.error, ""])

        return "\n".join(lines).strip() + "\n"


def dedupe_destinations(destinations: Iterable[Destination]) -> List[Destination]:
    """Remove duplicate destination names while preserving order."""

    seen = set()
    unique = []
    for destination in destinations:
        key = (destination.name.lower(), destination.country.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append(destination)
    return unique

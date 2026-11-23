"""Prompt builders for VoyagerAI's specialist agents."""

from __future__ import annotations

from dataclasses import dataclass

from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.utils import truncate

# Per-agent cap on research text handed to the itinerary agent.
RESEARCH_CONTEXT_CHARS = 12000
# Smallest reply budget for the itinerary agent, in tokens.
ITINERARY_MAX_TOKENS = 4200
# Reply tokens allowed per itinerary day, plus a fixed share for tips and budget.
ITINERARY_TOKENS_PER_DAY = 260
ITINERARY_FIXED_TOKENS = 1200
# Trips longer than this get an instruction to keep each day short.
LONG_TRIP_DAYS = 10
# Floor on research per agent, in characters.
MIN_RESEARCH_CHARS = 1100
# Tokens kept for the itinerary instructions.
INSTRUCTION_TOKENS = 512


def itinerary_max_tokens(days: int) -> int:
    """Reply budget that leaves room for every day plus the tips and budget sections."""

    return max(ITINERARY_MAX_TOKENS, days * ITINERARY_TOKENS_PER_DAY + ITINERARY_FIXED_TOKENS)


def itinerary_tokens_for_context(days: int, num_ctx: int) -> int:
    """Reply budget for a trip of `days` days that still fits a context of `num_ctx` tokens.

    Never below ITINERARY_MAX_TOKENS, so short trips behave as before. Longer trips grow
    the budget only as far as the window allows once the minimum research is in.
    """

    min_research_tokens = 4 * MIN_RESEARCH_CHARS // 3
    room = num_ctx - INSTRUCTION_TOKENS - min_research_tokens
    return max(ITINERARY_MAX_TOKENS, min(itinerary_max_tokens(days), room))


def research_chars_for_context(num_ctx: int, reply_tokens: int = ITINERARY_MAX_TOKENS) -> int:
    """Per-agent research budget, in characters, that fits a context of `num_ctx` tokens.

    The window has to hold four research reports, the instructions (about 512 tokens)
    and the reply. Three characters per token is a deliberately cautious estimate.
    """

    prompt_tokens = num_ctx - reply_tokens - INSTRUCTION_TOKENS
    return min(RESEARCH_CONTEXT_CHARS, max(MIN_RESEARCH_CHARS, prompt_tokens * 3 // 4))


DESTINATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "destinations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "country": {"type": "string"},
                    "flag": {"type": "string"},
                    "why": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3,
                    },
                },
                "required": ["name", "country", "why", "tags"],
            },
        }
    },
    "required": ["destinations"],
}


@dataclass
class AgentPrompt:
    system: str
    user: str
    temperature: float = 0.25
    max_tokens: int = 2800


def discovery_prompt(preferences: TravelPreferences, avoid: str = "") -> AgentPrompt:
    system = (
        "You are VoyagerAI's travel discovery expert. Suggest diverse destinations that fit "
        "seasonality, flight practicality, visa ease, trip length, and the user's preferred travel "
        "style. Return only JSON matching this shape: "
        '{"destinations":[{"name":"string","country":"string","flag":"string",'
        '"why":"one vivid specific sentence","tags":["short","tags"]}]}.'
    )
    avoid_line = f"Avoid repeating these destinations: {avoid}." if avoid else ""
    user = (
        f"Departure: {preferences.departure_city}\n"
        f"Nationality: {preferences.nationality_or_default}\n"
        f"Travel window: {preferences.months} months ({preferences.travel_window})\n"
        f"Trip length: {preferences.nights} nights / {preferences.days} days\n"
        f"Travel style: {preferences.vibe_description}\n"
        f"Flight preference: {preferences.flight_preference}\n"
        f"{avoid_line}\n"
        "Suggest exactly 8 worldwide destinations. Each suggestion should be plausible from "
        "the departure city, seasonally sensible within the travel window, and distinct from "
        "the others."
    )
    return AgentPrompt(system=system, user=user, temperature=0.35, max_tokens=1800)


def agent_prompt(
    agent_id: str,
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
    research_chars: int = RESEARCH_CONTEXT_CHARS,
    itinerary_tokens: int | None = None,
) -> AgentPrompt:
    if agent_id == "itinerary":
        return _itinerary_prompt(
            preferences,
            destination,
            search_context,
            prior_results,
            research_chars,
            itinerary_tokens,
        )
    builders = {
        "flight": _flight_prompt,
        "visa": _visa_prompt,
        "hotel": _hotel_prompt,
        "experience": _experience_prompt,
    }
    return builders[agent_id](preferences, destination, search_context, prior_results)


def _flight_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
) -> AgentPrompt:
    system = (
        "You are Flight Scout, a flight research specialist. Produce realistic route options, "
        "airlines, timing expectations, fare bands, and search tactics. Use ## headings. "
        "Do not invent exact live fares; provide ranges and source-backed caveats."
    )
    user = (
        f"Find the best {preferences.flight_preference} from {preferences.departure_city} to "
        f"{destination.display_name}. Window: {preferences.travel_window}. "
        f"Trip: {preferences.nights} nights. Nationality: "
        f"{preferences.nationality_or_default}. Travel style: {preferences.vibe_description}.\n\n"
        f"{search_context}\n\n"
        "Include: likely gateways, top airlines, economy and business fare ranges in USD, "
        "duration ranges, cheapest booking window, seasonal price variation, baggage or "
        "schedule watch-outs, and practical Google Flights / Skyscanner search links or "
        "query tips."
    )
    return AgentPrompt(system=system, user=user, temperature=0.2, max_tokens=2400)


def _visa_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
) -> AgentPrompt:
    system = (
        "You are Visa Advisor, a careful visa and immigration specialist. Prioritise official "
        "government sources. If a requirement is not clearly supported, say it needs verification. "
        "Use ## headings and end with a short 'Verify Before Booking' checklist."
    )
    user = (
        f"Visa and entry requirements for a tourist whose nationality is "
        f"{preferences.nationality_or_default}, visiting {destination.display_name}.\n\n"
        f"{search_context}\n\n"
        "Cover: visa-free / e-visa / visa-on-arrival status, max stay, passport validity, "
        "proof of funds or onward travel, health or insurance requirements, fees, processing "
        "time, recent rule changes if found, official government links, and uncertainty notes."
    )
    return AgentPrompt(system=system, user=user, temperature=0.1, max_tokens=2600)


def _hotel_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
) -> AgentPrompt:
    system = (
        "You are Hotel Finder, a practical accommodation researcher. Compare neighbourhoods and "
        "shortlist hotels by budget tier. Use ## headings and call out uncertainty in rates."
    )
    user = (
        f"Best hotels and neighbourhoods in {destination.display_name} for a "
        f"{preferences.nights}-night trip focused on {preferences.vibe_description}.\n\n"
        f"{search_context}\n\n"
        "Give three tiers: Budget, Mid-range, Luxury. For each tier: 2-3 hotel names or "
        "hotel-like areas if exact live availability is unavailable, neighbourhood, "
        "transit/access notes, standout feature, estimated nightly range in USD, and one "
        "booking tip."
    )
    return AgentPrompt(system=system, user=user, temperature=0.25, max_tokens=2600)


def _experience_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
) -> AgentPrompt:
    system = (
        f"You are Experience Curator, a tours and activities specialist. Focus on "
        f"{preferences.vibe_description}. Use ## headings and prefer experiences that fit the "
        f"trip length without overpacking."
    )
    user = (
        f"Top 7-8 tours and experiences in or around {destination.display_name} for a "
        f"{preferences.nights}-night trip.\n\n"
        f"{search_context}\n\n"
        "For each: name, type, duration, estimated price per person, best time of day, "
        "booking lead time, what makes it unmissable, and who should skip it. Include a "
        "balanced mix of anchors, local texture, and one flexible bad-weather option."
    )
    return AgentPrompt(system=system, user=user, temperature=0.35, max_tokens=2800)


def _itinerary_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    search_context: str,
    prior_results: dict[str, str],
    research_chars: int = RESEARCH_CONTEXT_CHARS,
    max_tokens: int | None = None,
) -> AgentPrompt:
    system = (
        "You are Itinerary Architect, a master travel planner. Write a vivid, practical, "
        "day-by-day itinerary. Use ## Day N - Title for every day. Each day must have Morning, "
        "Afternoon, and Evening subsections with named venues or areas. End with ## Practical Tips "
        "and ## Budget Breakdown sections."
    )
    prior = {
        key: truncate(value, research_chars)
        for key, value in prior_results.items()
        if key in {"flight", "visa", "hotel", "experience"}
    }
    user = (
        "Create a {nights}-night itinerary ({days} days) for {origin} to {destination}.\n"
        "Travel style: {vibe}\n"
        "Flight preference: {flight_pref}\n"
        "Nationality: {nationality}\n"
        "Window: {window}\n\n"
        "Flight context:\n{flight}\n\n"
        "Visa context:\n{visa}\n\n"
        "Hotel context:\n{hotel}\n\n"
        "Experience context:\n{experience}\n\n"
        "Structure exactly as:\n"
        "## Day 1 - Arrival & [neighbourhood]\n"
        "## Day 2 - [Theme]\n"
        "... all {days} days\n"
        "## Practical Tips\n"
        "## Budget Breakdown\n\n"
        "For Budget Breakdown include return economy flights, mid-range hotels for "
        "{nights} nights, food per day, activities total, local transport, and grand total "
        "range in USD. Keep the schedule realistic and include rest/buffer time."
        "{brevity}"
    ).format(
        nights=preferences.nights,
        days=preferences.days,
        origin=preferences.departure_city,
        destination=destination.display_name,
        vibe=preferences.vibe_description,
        flight_pref=preferences.flight_preference,
        nationality=preferences.nationality_or_default,
        window=preferences.travel_window,
        flight=prior.get("flight", "No flight research was available."),
        visa=prior.get("visa", "No visa research was available."),
        hotel=prior.get("hotel", "No hotel research was available."),
        experience=prior.get("experience", "No experience research was available."),
        brevity=(
            "\nThis is a long trip. Keep each Morning, Afternoon and Evening to one or two "
            "sentences so that every day, the Practical Tips and the Budget Breakdown fit."
            if preferences.days > LONG_TRIP_DAYS
            else ""
        ),
    )
    return AgentPrompt(
        system=system,
        user=user,
        temperature=0.35,
        max_tokens=max_tokens or itinerary_max_tokens(preferences.days),
    )

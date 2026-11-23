"""Prompt builders for VoyagerAI's specialist agents."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from voyager_ai.models import Destination, TravelPreferences
from voyager_ai.schemas import DESTINATION_JSON_SCHEMA  # noqa: F401  (re-exported)
from voyager_ai.structured import ResearchReport
from voyager_ai.utils import truncate

# Per-agent cap on research text handed to the itinerary agent.
RESEARCH_CONTEXT_CHARS = 12000
# Smallest reply budget for the itinerary agent, in tokens.
ITINERARY_MAX_TOKENS = 4200
# Reply tokens allowed per itinerary day, plus a fixed share for tips and budget.
ITINERARY_TOKENS_PER_DAY = 320
ITINERARY_FIXED_TOKENS = 1200
# Trips longer than this get an instruction to keep each day short.
LONG_TRIP_DAYS = 10
# Floor on research per agent, in characters.
MIN_RESEARCH_CHARS = 1100
# Tokens kept for the itinerary instructions.
INSTRUCTION_TOKENS = 512


# What each research agent covers.
_TOPICS = {
    "flight": (
        "likely gateways, top airlines, economy and business fare ranges in USD, "
        "duration ranges, cheapest booking window, seasonal price variation, baggage or "
        "schedule watch-outs, and practical Google Flights / Skyscanner search links or "
        "query tips"
    ),
    "visa": (
        "visa-free / e-visa / visa-on-arrival status, max stay, passport validity, "
        "proof of funds or onward travel, health or insurance requirements, fees, processing "
        "time, recent rule changes if found, official government links, and uncertainty notes"
    ),
    "hotel": (
        "2-3 hotel names or hotel-like areas if exact live availability is unavailable, "
        "neighbourhood, transit/access notes, standout feature, estimated nightly range in USD, "
        "and one booking tip"
    ),
    "experience": (
        "name, type, duration, estimated price per person, best time of day, "
        "booking lead time, what makes it unmissable, and who should skip it"
    ),
}
# (temperature, max_tokens) per research agent.
_SAMPLING = {
    "flight": (0.2, 2400),
    "visa": (0.1, 2600),
    "hotel": (0.25, 2600),
    "experience": (0.35, 2800),
}
_ROLES = {
    "flight": (
        "You are Flight Scout, a flight research specialist. Do not invent exact live fares; "
        "give ranges and source-backed caveats."
    ),
    "visa": (
        "You are Visa Advisor, a careful visa and immigration specialist. Prioritise official "
        "government sources. If a requirement is not clearly supported, say it needs "
        "verification."
    ),
    "hotel": (
        "You are Hotel Finder, a practical accommodation researcher. Compare neighbourhoods "
        "and shortlist hotels by budget tier, and call out uncertainty in rates."
    ),
    "experience": (
        "You are Experience Curator, a tours and activities specialist. Focus on {vibe}. "
        "Prefer experiences that fit the trip length without overpacking."
    ),
}
_JSON_RULE = "Return only JSON that matches the schema. Put each fact in its own finding."
_RESEARCH_AGENTS = ("flight", "visa", "hotel", "experience")


def _lead(agent_id: str, preferences: TravelPreferences, destination: Destination) -> str:
    """The opening request for one research agent."""

    if agent_id == "flight":
        return (
            f"Find the best {preferences.flight_preference} from {preferences.departure_city} to "
            f"{destination.display_name}. Window: {preferences.travel_window}. "
            f"Trip: {preferences.nights} nights. Nationality: "
            f"{preferences.nationality_or_default}. Travel style: {preferences.vibe_description}."
        )
    if agent_id == "visa":
        return (
            f"Visa and entry requirements for a tourist whose nationality is "
            f"{preferences.nationality_or_default}, visiting {destination.display_name}."
        )
    if agent_id == "hotel":
        return (
            f"Best hotels and neighbourhoods in {destination.display_name} for a "
            f"{preferences.nights}-night trip focused on {preferences.vibe_description}."
        )
    return (
        f"Top 7-8 tours and experiences in or around {destination.display_name} for a "
        f"{preferences.nights}-night trip."
    )


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


def research_prompt(
    agent_id: str,
    preferences: TravelPreferences,
    destination: Destination,
    context: str,
) -> AgentPrompt:
    """Prompt for one structured research agent (flight, visa, hotel or experience)."""

    role = _ROLES[agent_id].format(vibe=preferences.vibe_description)
    system = f"{role} Cover: {_TOPICS[agent_id]}. {_JSON_RULE}"
    parts = [_lead(agent_id, preferences, destination)]
    if context:
        parts.append(context)
    temperature, max_tokens = _SAMPLING[agent_id]
    return AgentPrompt(
        system=system,
        user="\n\n".join(parts),
        temperature=temperature,
        max_tokens=max_tokens,
    )


def render_report_for_prompt(report: ResearchReport | None, missing: str) -> str:
    """A research report as plain text for the itinerary prompt, or `missing` if absent."""

    if report is None:
        return missing
    lines = [report.summary]
    for finding in report.findings:
        if finding.basis == "sourced" and finding.source_url:
            lines.append(f"- [sourced] {finding.claim} ({finding.source_url})")
        else:
            lines.append(f"- [model] {finding.claim}")
    return "\n".join(lines)


def itinerary_prompt(
    preferences: TravelPreferences,
    destination: Destination,
    reports: Mapping[str, ResearchReport | None],
    research_chars: int,
    max_tokens: int,
) -> AgentPrompt:
    """Prompt for the structured itinerary agent, built from the research reports."""

    system = (
        "You are Itinerary Architect, a master travel planner. Plan a vivid, practical, "
        "day-by-day itinerary with named venues or areas in the morning, afternoon and evening "
        "of every day. Return only JSON that matches the schema."
    )
    research = {
        agent_id: truncate(
            render_report_for_prompt(
                reports.get(agent_id), f"No {agent_id} research was available."
            ),
            research_chars,
        )
        for agent_id in _RESEARCH_AGENTS
    }
    brevity = (
        "\nThis is a long trip. Keep each morning, afternoon and evening to one or two "
        "sentences so that every day, the tips and the budget fit."
        if preferences.days > LONG_TRIP_DAYS
        else ""
    )
    user = (
        f"Create a {preferences.nights}-night itinerary ({preferences.days} days) for "
        f"{preferences.departure_city} to {destination.display_name}.\n"
        f"Travel style: {preferences.vibe_description}\n"
        f"Flight preference: {preferences.flight_preference}\n"
        f"Nationality: {preferences.nationality_or_default}\n"
        f"Window: {preferences.travel_window}\n\n"
        f"Flight context:\n{research['flight']}\n\n"
        f"Visa context:\n{research['visa']}\n\n"
        f"Hotel context:\n{research['hotel']}\n\n"
        f"Experience context:\n{research['experience']}\n\n"
        f"Include all {preferences.days} days, practical tips and a budget. The budget lines "
        f"are: Return economy flights, Mid-range hotel ({preferences.nights} nights), Food, "
        "Activities and Local transport, all in USD, plus a grand total range. Keep the "
        f"schedule realistic and include rest/buffer time.{brevity}"
    )
    return AgentPrompt(system=system, user=user, temperature=0.35, max_tokens=max_tokens)

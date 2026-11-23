"""Render a structured trip plan as Markdown.

Claims are never shown as fact without a source: a finding without a fetched source is
labelled as model knowledge, and visa rules always carry a reminder to check the official site.
"""

from __future__ import annotations

from voyager_ai.links import NoPrices, PriceProvider, deep_links
from voyager_ai.models import AgentResult, TripPlan
from voyager_ai.structured import CostRange, Finding, Itinerary, Slot

UNSOURCED_LABEL = "unverified: model knowledge"
VISA_NOTE = "Check these rules on the official government site before you book."

_RESEARCH_AGENTS = ("flight", "visa", "hotel", "experience")
# Which booking links appear under which research section.
SECTION_LINKS = {"flight": ("Google Flights", "Skyscanner"), "hotel": ("Booking.com",)}


def money(amount: float) -> str:
    return f"{round(amount):,}"


def cost_range(cost: CostRange) -> str:
    return f"USD {money(cost.low)}-{money(cost.high)}"


def slot_line(name: str, slot: Slot) -> str:
    text = slot.title
    if slot.description and slot.description != slot.title:
        text += f": {slot.description}"
    if slot.place:
        text += f" ({slot.place})"
    if slot.cost:
        text += f" - {cost_range(slot.cost)}"
    return f"**{name}** - {text}"


def finding_line(finding: Finding) -> str:
    lead = f"**{finding.topic}:** " if finding.topic else ""
    if finding.basis == "sourced" and finding.source_url:
        return f"- {lead}{finding.claim} ([source]({finding.source_url}))"
    return f"- {lead}{finding.claim} _({UNSOURCED_LABEL})_"


def _notices(result: AgentResult) -> list[str]:
    lines = [f"> Warning: {warning}\n" for warning in result.warnings]
    if result.error:
        lines.append(f"> Agent error: {result.error}\n")
    return lines


def _facts(plan: TripPlan) -> list[str]:
    prefs = plan.preferences
    return [
        f"# VoyagerAI Trip Plan: {plan.destination.display_name}\n",
        f"- Departure: {prefs.departure_city}",
        f"- Nationality: {prefs.nationality_or_default}",
        f"- Travel window: {prefs.travel_window}",
        f"- Trip length: {prefs.nights} nights / {prefs.days} days",
        f"- Flight preference: {prefs.flight_preference}",
        f"- Travel style: {prefs.vibe_description}\n",
    ]


def _itinerary(itinerary: Itinerary) -> list[str]:
    lines = ["## Itinerary\n"]
    for day in itinerary.days:
        lines.append(f"### Day {day.number} - {day.title}\n")
        # A blank line after each slot keeps them as separate paragraphs.
        lines.append(slot_line("Morning", day.morning) + "\n")
        lines.append(slot_line("Afternoon", day.afternoon) + "\n")
        lines.append(slot_line("Evening", day.evening) + "\n")
    if itinerary.tips:
        lines.append("## Practical Tips\n")
        lines.extend(f"- {tip}" for tip in itinerary.tips)
        lines.append("")
    budget = itinerary.budget
    if budget.items:
        lines.append("## Budget\n")
        lines.append(f"| Item | Low ({budget.currency}) | High ({budget.currency}) |")
        lines.append("| --- | ---: | ---: |")
        for item in budget.items:
            lines.append(f"| {item.label} | {money(item.cost.low)} | {money(item.cost.high)} |")
        total = budget.total
        lines.append(f"| **Total** | {money(total.low)} | {money(total.high)} |\n")
    lines.extend(f"> Warning: {warning}\n" for warning in itinerary.warnings)
    return lines


def _research(plan: TripPlan, result: AgentResult, prices: PriceProvider) -> list[str]:
    lines = [f"## {result.label}\n"]
    if result.agent_id == "visa":
        lines.append(f"{VISA_NOTE}\n")
    if result.report:
        lines.append(f"{result.report.summary}\n")
        if result.report.findings:
            lines.extend(finding_line(finding) for finding in result.report.findings)
            lines.append("")
    names = SECTION_LINKS.get(result.agent_id, ())
    if names:
        links = deep_links(plan.preferences, plan.destination)
        lines.append("### Booking Links\n")
        lines.extend(f"- [{name}]({links[name]})" for name in names)
        if result.agent_id == "flight":
            for quote in prices.quotes(plan.preferences, plan.destination):
                lines.append(f"- [{quote.label}]({quote.url}) - {cost_range(quote.cost)}")
        lines.append("")
    if result.sources:
        lines.append("### Sources Considered\n")
        lines.extend(source.to_markdown() for source in result.sources)
        lines.append("")
    lines.extend(_notices(result))
    return lines


def render_markdown(plan: TripPlan, prices: PriceProvider | None = None) -> str:
    """The whole plan as Markdown: trip facts, itinerary, then each research section."""

    prices = prices or NoPrices()
    lines = _facts(plan)
    final = plan.results.get("itinerary")
    if final is not None:
        if final.itinerary is not None:
            lines.extend(_itinerary(final.itinerary))
        else:
            lines.append("## Itinerary\n")
            lines.append("No itinerary could be built for this run.\n")
        lines.extend(_notices(final))
    for agent_id in _RESEARCH_AGENTS:
        result = plan.results.get(agent_id)
        if result is not None:
            lines.extend(_research(plan, result, prices))
    return "\n".join(lines).strip() + "\n"

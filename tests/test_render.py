from voyager_ai.links import NoPrices, PriceQuote, deep_links
from voyager_ai.models import AgentResult, Destination, SearchResult, TravelPreferences, TripPlan
from voyager_ai.render import (
    ESTIMATE_NOTE,
    UNSOURCED_LABEL,
    VISA_NOTE,
    md_escape,
    render_markdown,
)
from voyager_ai.structured import (
    Budget,
    BudgetItem,
    CostRange,
    DayPlan,
    Finding,
    Itinerary,
    ResearchReport,
    Slot,
)

DEST = Destination("Tokyo", "Japan", "x")
PREFS = TravelPreferences("Mumbai")


def _itinerary(items=(), warnings=()):
    day = DayPlan(
        1,
        "Arrival",
        Slot("Shrine walk", "Early visit.", "Meiji Jingu", CostRange(0, 10)),
        Slot("Lunch", "", None, None),
        Slot("Izakaya", "Small plates.", None, CostRange(1200, 2500)),
    )
    return Itinerary(
        days=[day],
        tips=["Buy a Suica card."],
        budget=Budget(items=list(items)),
        warnings=list(warnings),
    )


def _trip(results):
    return TripPlan(preferences=PREFS, destination=DEST, results=results)


def _visa_result():
    report = ResearchReport(
        "Visa summary.",
        [
            Finding("Entry", "Visa-free.", "sourced", "https://www.gov.uk/x"),
            Finding("Fees", "No fee.", "model"),
        ],
    )
    return AgentResult(
        "visa",
        model="m",
        report=report,
        sources=[SearchResult("Gov", "https://www.gov.uk/x", "rules")],
        warnings=["Search failed for 'q': down"],
    )


def test_markdown_labels_unsourced_findings_and_adds_visa_note():
    md = render_markdown(_trip({"visa": _visa_result()}))
    assert f"- **Fees:** No fee. _({UNSOURCED_LABEL})_" in md
    assert "- **Entry:** Visa-free. ([source](https://www.gov.uk/x))" in md
    assert f"## Visa Advisor\n\n{VISA_NOTE}" in md
    assert "### Sources Considered" in md and "> Warning: Search failed" in md


def test_budget_table_has_computed_total():
    items = [
        BudgetItem("Flights", CostRange(800, 1200)),
        BudgetItem("Hotel", CostRange(700, 1400)),
    ]
    itinerary = AgentResult("itinerary", model="m", itinerary=_itinerary(items))
    md = render_markdown(_trip({"itinerary": itinerary}))
    assert "| Item | Low (USD) | High (USD) |" in md
    assert "| Flights | 800 | 1,200 |" in md
    assert "| **Total** | 1,500 | 2,600 |" in md


def test_itinerary_days_tips_and_warnings_in_order():
    itinerary = AgentResult(
        "itinerary",
        model="m",
        itinerary=_itinerary([BudgetItem("Food", CostRange(10, 20))], ["Day 2 is missing."]),
    )
    md = render_markdown(_trip({"itinerary": itinerary, "visa": _visa_result()}))
    assert "### Day 1 - Arrival" in md
    assert "**Morning** - Shrine walk: Early visit. (Meiji Jingu) - USD 0-10" in md
    assert "**Afternoon** - Lunch\n" in md
    assert "**Evening** - Izakaya: Small plates. - USD 1,200-2,500" in md
    order = [
        md.index(h)
        for h in (
            "## Itinerary",
            "## Practical Tips",
            "## Budget",
            "> Warning: Day 2 is missing.",
            "## Visa Advisor",
        )
    ]
    assert order == sorted(order)


def test_agent_errors_and_missing_itinerary_are_reported():
    failed = AgentResult("itinerary", model="m", status="error", error="bad json")
    md = render_markdown(_trip({"itinerary": failed}))
    assert md.startswith("# VoyagerAI Trip Plan: Tokyo, Japan")
    assert "## Itinerary" in md and "> Agent error: bad json" in md


def test_booking_links_under_flight_and_hotel():
    report = ResearchReport("S", [])
    md = render_markdown(
        _trip(
            {
                "flight": AgentResult("flight", model="m", report=report),
                "hotel": AgentResult("hotel", model="m", report=report),
            }
        )
    )
    links = deep_links(PREFS, DEST)
    flight = md[md.index("## Flight Scout") : md.index("## Hotel Finder")]
    hotel = md[md.index("## Hotel Finder") :]
    assert f"[Google Flights]({links['Google Flights']})" in flight
    assert f"[Skyscanner]({links['Skyscanner']})" in flight
    assert f"[Booking.com]({links['Booking.com']})" in hotel


def test_price_quotes_are_printed_when_a_provider_has_them():
    class OneQuote:
        def quotes(self, preferences, destination):
            return [PriceQuote("Economy return", CostRange(650, 980), "https://fares.example/q")]

    flight = AgentResult("flight", model="m", report=ResearchReport("S", []))
    md = render_markdown(_trip({"flight": flight}), prices=OneQuote())
    assert "[Economy return](https://fares.example/q) - USD 650-980" in md
    assert NoPrices().quotes(PREFS, DEST) == []
    assert "fares.example" not in render_markdown(_trip({"flight": flight}))


def test_deep_links():
    links = deep_links(TravelPreferences("Mumbai"), Destination("Tokyo", "Japan", "x"))
    assert links["Google Flights"] == (
        "https://www.google.com/travel/flights?q=Flights%20from%20Mumbai%20to%20Tokyo%2C%20Japan"
    )
    assert links["Skyscanner"] == "https://www.skyscanner.net/"
    assert links["Booking.com"] == "https://www.booking.com/searchresults.html?ss=Tokyo%2C%20Japan"


def test_estimate_note_sits_under_itinerary_and_budget():
    itinerary = AgentResult(
        "itinerary", model="m", itinerary=_itinerary([BudgetItem("Food", CostRange(10, 20))])
    )
    md = render_markdown(_trip({"itinerary": itinerary}))
    assert f"## Itinerary\n\n{ESTIMATE_NOTE}\n" in md
    assert f"## Budget\n\n{ESTIMATE_NOTE}\n" in md


def test_md_escape():
    assert md_escape("a [b](c) | *d* _e_ `f` <g> #h $1 \\") == (
        "a \\[b\\]\\(c\\) \\| \\*d\\* \\_e\\_ \\`f\\` \\<g\\> \\#h \\$1 \\\\"
    )
    assert md_escape("one\n\n  two\r\nthree") == "one two three"


def _report_trip(finding):
    report = ResearchReport("Summary.", [finding])
    return _trip({"visa": AgentResult("visa", model="m", report=report)})


def test_injected_link_in_a_claim_is_not_a_link():
    finding = Finding("Entry", "See [official site](https://evil.example) now.", "model")
    md = render_markdown(_report_trip(finding))
    assert "\\[official site\\]" in md
    assert "](https://evil.example)" not in md


def test_newline_in_a_claim_does_not_start_a_heading():
    finding = Finding("Entry", "Fine.\n# Pwned", "model")
    md = render_markdown(_report_trip(finding))
    assert "\n# Pwned" not in md
    assert "Fine. \\# Pwned" in md


def test_pipe_in_a_budget_label_keeps_the_table_shape():
    items = [BudgetItem("Food | drinks", CostRange(10, 20))]
    itinerary = AgentResult("itinerary", model="m", itinerary=_itinerary(items))
    md = render_markdown(_trip({"itinerary": itinerary}))
    row = next(line for line in md.splitlines() if line.startswith("| Food"))
    header = next(line for line in md.splitlines() if line.startswith("| Item"))
    assert row.replace("\\|", "").count("|") == header.count("|")


def test_dollar_amounts_are_not_latex():
    itinerary = _itinerary()
    itinerary.days[0].morning = Slot("Market", "$50 to $80 per person.", None, None)
    md = render_markdown(
        _trip({"itinerary": AgentResult("itinerary", model="m", itinerary=itinerary)})
    )
    assert "\\$50 to \\$80" in md


def test_source_titles_are_escaped_and_urls_are_safe():
    result = AgentResult(
        "hotel",
        model="m",
        report=ResearchReport("S", []),
        sources=[SearchResult("[Click](https://evil.example)", "https://a.example/x y)", "s")],
    )
    md = render_markdown(_trip({"hotel": result}))
    assert "](https://evil.example)" not in md
    assert "(https://a.example/x%20y%29)" in md

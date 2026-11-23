from __future__ import annotations

from typing import Any

import pytest

from voyager_ai.models import ValidationError
from voyager_ai.structured import (
    Budget,
    BudgetItem,
    CostRange,
    Itinerary,
    ResearchReport,
    Slot,
    normalize_url,
)

ALLOWED = ["https://www.mofa.go.jp/j_info/visit/visa/"]


def _report(**finding: Any) -> ResearchReport:
    base = {
        "topic": "Entry",
        "claim": "Visa-free for 90 days.",
        "basis": "sourced",
        "source_url": None,
    }
    return ResearchReport.from_payload({"summary": "S", "findings": [{**base, **finding}]}, ALLOWED)


def _slot(name: str) -> dict[str, Any]:
    return {"title": f"{name} title", "description": f"{name} text", "place": None, "cost": None}


def _itin(numbers: list[int], drop: str | None = None) -> dict[str, Any]:
    days = []
    for n in numbers:
        day: dict[str, Any] = {"number": n, "title": f"Day titled {n}"}
        for name in ("morning", "afternoon", "evening"):
            if name != drop:
                day[name] = _slot(name)
        days.append(day)
    return {"days": days, "tips": [], "budget": {"items": []}}


def test_known_url_is_sourced_and_keeps_allowed_spelling():
    f = _report(source_url="HTTPS://www.MOFA.go.jp/j_info/visit/visa").findings[0]
    assert f.basis == "sourced" and f.source_url == ALLOWED[0]


def test_invented_url_is_downgraded():
    f = _report(source_url="https://made-up.example/visa").findings[0]
    assert f.basis == "model" and f.source_url is None


def test_model_claim_with_known_url_stays_model():
    f = _report(basis="model", source_url=ALLOWED[0]).findings[0]
    assert f.basis == "model" and f.source_url is None


def test_unknown_basis_with_known_url_is_model():
    f = _report(basis="maybe", source_url=ALLOWED[0]).findings[0]
    assert f.basis == "model" and f.source_url is None


def test_finding_without_claim_is_dropped():
    assert _report(claim="  ").findings == []


def test_report_requires_summary():
    with pytest.raises(ValidationError, match="no summary"):
        ResearchReport.from_payload({"summary": "", "findings": []}, [])


def test_sourced_count_counts_only_sourced():
    report = _report(source_url=ALLOWED[0])
    assert report.sourced_count == 1
    assert _report(source_url="https://nope.example").sourced_count == 0


def test_normalize_url():
    assert normalize_url("HTTPS://Example.COM/Path/#frag") == "https://example.com/Path"
    assert normalize_url("https://example.com/") == "https://example.com/"


def test_itinerary_sorts_renumbers_and_truncates():
    it = Itinerary.from_payload(_itin(numbers=[2, 1, 3]), expected_days=2)
    assert [d.number for d in it.days] == [1, 2] and it.days[0].title == "Day titled 1"


def test_short_itinerary_warns():
    it = Itinerary.from_payload(_itin(numbers=[1, 2, 3]), expected_days=5)
    assert "The itinerary covers 3 of 5 days." in it.warnings


def test_missing_slot_becomes_free_time():
    it = Itinerary.from_payload(_itin(numbers=[1], drop="evening"), expected_days=1)
    assert it.days[0].evening == Slot("Free time", "Nothing planned.")
    assert "Day 1 had no evening plan." in it.warnings


def test_invalid_slot_cost_becomes_none():
    payload = _itin(numbers=[1])
    payload["days"][0]["morning"]["cost"] = {"low": -5, "high": 2}
    assert Itinerary.from_payload(payload, 1).days[0].morning.cost is None


def test_budget_total_is_computed():
    budget = Budget(
        [BudgetItem("Flights", CostRange(800, 1200)), BudgetItem("Hotels", CostRange(700, 1400))]
    )
    assert budget.total == CostRange(1500, 2600)


def test_invalid_budget_items_are_skipped():
    payload = _itin(numbers=[1])
    payload["budget"] = {
        "items": [{"label": "Food", "low": 10, "high": 20}, {"label": "Bad", "low": -1, "high": 2}]
    }
    assert [i.label for i in Itinerary.from_payload(payload, 1).budget.items] == ["Food"]


def test_cost_range_swaps_and_rejects_negative():
    assert CostRange.from_payload({"low": 50, "high": 20}) == CostRange(20, 50)
    with pytest.raises(ValidationError):
        CostRange.from_payload({"low": -1, "high": 5})
    with pytest.raises(ValidationError):
        CostRange.from_payload({"low": "cheap", "high": 5})


def test_itinerary_without_days_is_rejected():
    with pytest.raises(ValidationError, match="no days"):
        Itinerary.from_payload({"days": [], "tips": [], "budget": {"items": []}}, 3)


def test_malformed_url_is_downgraded_not_fatal():
    f = _report(source_url="http://[bad").findings[0]
    assert f.basis == "model" and f.source_url is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_cost_range_rejects_non_finite(bad):
    with pytest.raises(ValidationError):
        CostRange.from_payload({"low": bad, "high": 5})
    with pytest.raises(ValidationError):
        CostRange.from_payload({"low": 1, "high": bad})


def test_root_slash_is_equivalent():
    assert normalize_url("https://example.com") == normalize_url("https://example.com/")
    f = ResearchReport.from_payload(
        {
            "summary": "S",
            "findings": [
                {
                    "topic": "t",
                    "claim": "c",
                    "basis": "sourced",
                    "source_url": "https://example.com",
                }
            ],
        },
        ["https://example.com/"],
    ).findings[0]
    assert f.basis == "sourced" and f.source_url == "https://example.com/"


def test_sourced_without_url_becomes_model():
    f = _report(basis="sourced", source_url=None).findings[0]
    assert f.basis == "model" and f.source_url is None

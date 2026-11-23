"""Validated structured types for research reports and itineraries.

A finding is "sourced" only when its URL is one the app actually fetched. Anything else is
labelled as model knowledge, so the report never presents an invented link as evidence.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from voyager_ai.models import ValidationError
from voyager_ai.utils import clean_text

Basis = Literal["sourced", "model"]

SLOT_NAMES = ("morning", "afternoon", "evening")


def normalize_url(url: str) -> str:
    """Lowercase scheme and host, drop the fragment and any trailing slash below the root."""

    cleaned = clean_text(url)
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return cleaned
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    if not path and parts.netloc:
        path = "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


@dataclass
class Finding:
    topic: str
    claim: str
    basis: Basis
    source_url: str | None = None


@dataclass
class ResearchReport:
    summary: str
    findings: list[Finding] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any, allowed_urls: Collection[str]) -> ResearchReport:
        data = payload if isinstance(payload, dict) else {}
        summary = clean_text(data.get("summary"))
        if not summary:
            raise ValidationError("The research report has no summary.")

        allowed = {normalize_url(url): url for url in allowed_urls}
        raw = data.get("findings")
        findings: list[Finding] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            claim = clean_text(item.get("claim"))
            if not claim:
                continue
            url = clean_text(item.get("source_url"))
            matched = allowed.get(normalize_url(url)) if url else None
            findings.append(
                Finding(
                    topic=clean_text(item.get("topic")),
                    claim=claim,
                    basis="sourced" if matched else "model",
                    source_url=matched,
                )
            )
        return cls(summary=summary, findings=findings)

    @property
    def sourced_count(self) -> int:
        return sum(1 for finding in self.findings if finding.basis == "sourced")


@dataclass
class CostRange:
    low: float
    high: float

    @classmethod
    def from_payload(cls, payload: Any) -> CostRange:
        if not isinstance(payload, dict):
            raise ValidationError("A cost needs low and high amounts.")
        low, high = payload.get("low"), payload.get("high")
        for value in (low, high):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError("Cost amounts must be numbers.")
            if not math.isfinite(value):
                raise ValidationError("Cost amounts must be finite numbers.")
            if value < 0:
                raise ValidationError("Cost amounts cannot be negative.")
        assert isinstance(low, (int, float)) and isinstance(high, (int, float))
        return cls(low=min(low, high), high=max(low, high))


@dataclass
class Slot:
    title: str
    description: str
    place: str | None = None
    cost: CostRange | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> Slot | None:
        if not isinstance(payload, dict):
            return None
        title = clean_text(payload.get("title"))
        description = clean_text(payload.get("description"))
        if not title and not description:
            return None
        try:
            cost: CostRange | None = CostRange.from_payload(payload.get("cost"))
        except ValidationError:
            cost = None
        return cls(
            title=title or description,
            description=description,
            place=clean_text(payload.get("place")) or None,
            cost=cost,
        )


@dataclass
class DayPlan:
    number: int
    title: str
    morning: Slot
    afternoon: Slot
    evening: Slot


@dataclass
class BudgetItem:
    label: str
    cost: CostRange


@dataclass
class Budget:
    items: list[BudgetItem] = field(default_factory=list)
    currency: str = "USD"

    @property
    def total(self) -> CostRange:
        return CostRange(
            low=sum(item.cost.low for item in self.items),
            high=sum(item.cost.high for item in self.items),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> Budget:
        raw = payload.get("items") if isinstance(payload, dict) else None
        items: list[BudgetItem] = []
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                continue
            label = clean_text(entry.get("label"))
            try:
                cost = CostRange.from_payload(entry)
            except ValidationError:
                continue
            if label:
                items.append(BudgetItem(label, cost))
        return cls(items=items)


def _day_number(day: dict[str, Any]) -> float:
    number = day.get("number")
    if isinstance(number, bool) or not isinstance(number, (int, float)):
        return float("inf")
    return number


@dataclass
class Itinerary:
    days: list[DayPlan]
    tips: list[str]
    budget: Budget
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any, expected_days: int) -> Itinerary:
        data = payload if isinstance(payload, dict) else {}
        raw_days = data.get("days")
        raw = [d for d in raw_days if isinstance(d, dict)] if isinstance(raw_days, list) else []
        if not raw:
            raise ValidationError("The itinerary has no days.")

        # sorted() is stable, so days without a usable number keep their order at the end.
        ordered = sorted(raw, key=_day_number)[:expected_days]
        warnings: list[str] = []
        days: list[DayPlan] = []
        for number, entry in enumerate(ordered, start=1):
            slots: dict[str, Slot] = {}
            for name in SLOT_NAMES:
                slot = Slot.from_payload(entry.get(name))
                if slot is None:
                    slot = Slot("Free time", "Nothing planned.")
                    warnings.append(f"Day {number} had no {name} plan.")
                slots[name] = slot
            days.append(
                DayPlan(
                    number=number,
                    title=clean_text(entry.get("title")) or f"Day {number}",
                    morning=slots["morning"],
                    afternoon=slots["afternoon"],
                    evening=slots["evening"],
                )
            )
        if len(days) < expected_days:
            warnings.insert(0, f"The itinerary covers {len(days)} of {expected_days} days.")

        raw_tips = data.get("tips")
        tips = (
            [t for t in (clean_text(x) for x in raw_tips) if t]
            if isinstance(raw_tips, list)
            else []
        )
        return cls(
            days=days,
            tips=tips,
            budget=Budget.from_payload(data.get("budget")),
            warnings=warnings,
        )

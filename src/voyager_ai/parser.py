"""Parsers for LLM JSON and itinerary markdown."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List

from voyager_ai.models import ValidationError


@dataclass
class ItineraryDay:
    num: int
    title: str
    body: str


@dataclass
class ItinerarySection:
    title: str
    body: str


@dataclass
class ParsedItinerary:
    days: List[ItineraryDay]
    extras: List[ItinerarySection]


DAY_RE = re.compile(r"^##\s+Day\s+(\d+)\s*[:\-\u2013\u2014]?\s*(.*)$", re.IGNORECASE)
HEADING_RE = re.compile(r"^##\s+(.*)$")


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _find_json_candidate(text: str) -> str:
    """Find the first balanced JSON object or array in a model response."""

    cleaned = strip_code_fence(text)
    starts = [index for index in (cleaned.find("{"), cleaned.find("[")) if index >= 0]
    if not starts:
        return cleaned

    start = min(starts)
    stack = []
    in_string = False
    escaped = False
    pairs = {"{": "}", "[": "]"}

    for index, char in enumerate(cleaned[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in ("}", "]"):
            if not stack or stack[-1] != char:
                break
            stack.pop()
            if not stack:
                return cleaned[start : index + 1]

    return cleaned[start:]


def extract_json(text: str) -> Any:
    candidate = _find_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValidationError("The model did not return valid JSON: %s" % exc) from exc


def parse_itinerary(text: str) -> ParsedItinerary:
    if not text:
        return ParsedItinerary(days=[], extras=[])

    chunks = [chunk for chunk in re.split(r"(?m)^(?=##\s+)", text) if chunk.strip()]
    days = []
    extras = []
    for chunk in chunks:
        stripped = chunk.strip()
        first_line = stripped.splitlines()[0]
        day_match = DAY_RE.match(first_line)
        if day_match:
            body = re.sub(r"^##[^\n]*\n?", "", stripped, count=1).strip()
            days.append(
                ItineraryDay(
                    num=int(day_match.group(1)),
                    title=day_match.group(2).strip() or "Day %s" % day_match.group(1),
                    body=body,
                )
            )
            continue

        heading_match = HEADING_RE.match(first_line)
        if heading_match:
            body = re.sub(r"^##[^\n]*\n?", "", stripped, count=1).strip()
            extras.append(ItinerarySection(title=heading_match.group(1).strip(), body=body))

    return ParsedItinerary(days=days, extras=extras)

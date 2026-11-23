from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, validate

from voyager_ai.demo import demo_itinerary_payload, demo_research_payload
from voyager_ai.schemas import DESTINATION_JSON_SCHEMA, ITINERARY_SCHEMA, RESEARCH_SCHEMA
from voyager_ai.structured import Itinerary, ResearchReport


def _load(name: str) -> Any:
    return json.loads((Path(__file__).parent / "fixtures" / name).read_text())


def _objects(node: Any):
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            yield node
        for value in node.values():
            yield from _objects(value)
    elif isinstance(node, list):
        for value in node:
            yield from _objects(value)


@pytest.mark.parametrize("schema", [DESTINATION_JSON_SCHEMA, RESEARCH_SCHEMA, ITINERARY_SCHEMA])
def test_schemas_are_valid(schema):
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("schema", [RESEARCH_SCHEMA, ITINERARY_SCHEMA])
def test_schemas_are_strict_and_portable(schema):
    text = json.dumps(schema)
    assert "oneOf" not in text and "anyOf" not in text and "$ref" not in text
    for obj in _objects(schema):
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_demo_payloads_match_schemas():
    for agent in ("flight", "visa", "hotel", "experience"):
        payload = demo_research_payload(agent, "Tokyo, Japan")
        validate(payload, RESEARCH_SCHEMA)
        assert len(payload["findings"]) == 3
        assert {f["basis"] for f in payload["findings"]} == {"model"}
    validate(demo_itinerary_payload(31), ITINERARY_SCHEMA)
    assert len(demo_itinerary_payload(31)["days"]) == 31
    small = demo_itinerary_payload(2)
    assert len(small["tips"]) == 2 and len(small["budget"]["items"]) == 5


def test_fixtures_match_schemas_and_parse_cleanly():
    visa = _load("visa_report.json")
    validate(visa["payload"], RESEARCH_SCHEMA)
    assert ResearchReport.from_payload(visa["payload"], visa["allowed_urls"]).sourced_count == 2
    itin = _load("itinerary_3_nights.json")
    validate(itin, ITINERARY_SCHEMA)
    assert Itinerary.from_payload(itin, expected_days=4).warnings == []

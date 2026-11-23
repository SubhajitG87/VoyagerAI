"""JSON schemas for structured LLM output.

Every object lists all of its properties in ``required`` and sets
``additionalProperties: false``. There is no ``oneOf``, ``anyOf`` or ``$ref``, so Ollama's
``format`` and vLLM's guided decoding both accept these schemas. Nullable fields use a
type list such as ``["string", "null"]``.
"""

from __future__ import annotations

from typing import Any

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


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


_COST: dict[str, Any] = _obj({"low": {"type": "number"}, "high": {"type": "number"}})
_COST_OR_NULL: dict[str, Any] = {**_COST, "type": ["object", "null"]}

_SLOT = _obj(
    {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "place": {"type": ["string", "null"]},
        "cost": _COST_OR_NULL,
    }
)

RESEARCH_SCHEMA: dict[str, Any] = _obj(
    {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "minItems": 0,
            "maxItems": 12,
            "items": _obj(
                {
                    "topic": {"type": "string"},
                    "claim": {"type": "string"},
                    "basis": {"type": "string", "enum": ["sourced", "model"]},
                    "source_url": {"type": ["string", "null"]},
                }
            ),
        },
    }
)

ITINERARY_SCHEMA: dict[str, Any] = _obj(
    {
        "days": {
            "type": "array",
            "items": _obj(
                {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "morning": _SLOT,
                    "afternoon": _SLOT,
                    "evening": _SLOT,
                }
            ),
        },
        "tips": {"type": "array", "items": {"type": "string"}},
        "budget": _obj(
            {
                "items": {
                    "type": "array",
                    "items": _obj(
                        {
                            "label": {"type": "string"},
                            "low": {"type": "number"},
                            "high": {"type": "number"},
                        }
                    ),
                }
            }
        ),
    }
)

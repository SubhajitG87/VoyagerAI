"""Probe which structured-output modes an OpenAI-compatible endpoint honours, per model.

Usage:
    NVIDIA_API_KEY=... python scripts/probe_schema.py --model <id> [--model <id> ...]
    NVIDIA_API_KEY=... python scripts/probe_schema.py --model <id> --schemas real

Prints one line per (model, mode, schema): PASS, or FAIL with a reason. PASS means HTTP 200 and
a reply that validates against the schema (needs the ``jsonschema`` package). ``--schemas real``
also sends the app's research and itinerary schemas, which are larger and nested.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, cast, get_args

from voyager_ai.config import NIM_BASE_URL
from voyager_ai.llm import LLMError, OpenAICompatibleClient, SchemaMode
from voyager_ai.schemas import ITINERARY_SCHEMA, RESEARCH_SCHEMA

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}
MODES: tuple[str, ...] = get_args(SchemaMode)


@dataclass(frozen=True)
class Probe:
    name: str
    schema: dict[str, Any]
    user: str
    max_tokens: int = 800


SIMPLE = Probe("simple", SCHEMA, "Name one city for a short trip and a number of days.")
REAL = (
    Probe(
        "research",
        RESEARCH_SCHEMA,
        "Visa and entry requirements for an Indian tourist visiting Tokyo, Japan. Give a short "
        'summary and two findings. No sources are available, so set basis to "model" and '
        "source_url to null.",
        1200,
    ),
    Probe(
        "itinerary",
        ITINERARY_SCHEMA,
        "Create a 1-night itinerary (2 days) from Mumbai to Tokyo, Japan with a morning, "
        "afternoon and evening each day, two practical tips and a budget in USD with Flights, "
        "Hotel and Food lines. Keep every description to one sentence.",
        2400,
    ),
)
SCHEMA_SETS = {"simple": (SIMPLE,), "real": (SIMPLE, *REAL)}


def validate(text: str, schema: dict[str, Any] = SCHEMA) -> str | None:
    """Return None when ``text`` is JSON matching ``schema``, else the reason it is not."""

    import jsonschema  # imported lazily so importing this script never needs it

    try:
        jsonschema.validate(json.loads(text), schema)
    except json.JSONDecodeError as exc:
        return f"not JSON ({exc.msg})"
    except jsonschema.ValidationError as exc:
        return f"schema violation ({exc.message})"
    return None


def probe(base_url: str, api_key: str, model: str, mode: str, check: Probe = SIMPLE) -> str:
    client = OpenAICompatibleClient(
        base_url,
        api_key,
        retries=1,
        schema_mode=cast(SchemaMode, mode),
        fallback=False,
    )
    try:
        response = client.complete(
            agent_id="probe",
            model=model,
            system="Reply with JSON only.",
            user=check.user,
            max_tokens=check.max_tokens,
            json_schema=check.schema,
        )
    except LLMError as exc:
        return f"FAIL {exc}"
    problem = validate(response.text, check.schema)
    if problem is not None and response.truncated:
        problem += "; the reply was cut off at the token limit"
    return "PASS" if problem is None else f"FAIL {problem}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--base-url", default=NIM_BASE_URL)
    parser.add_argument("--model", action="append", default=[], required=True)
    parser.add_argument("--mode", action="append", choices=MODES, default=None)
    parser.add_argument(
        "--schemas",
        choices=tuple(SCHEMA_SETS),
        default="simple",
        help="simple: a two-field schema; real: also the app's research and itinerary schemas",
    )
    args = parser.parse_args(argv)

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
    if not api_key:
        print("Set NVIDIA_API_KEY (or OPENAI_COMPATIBLE_API_KEY).", file=sys.stderr)
        return 2
    for model in args.model:
        for mode in args.mode or MODES:
            for check in SCHEMA_SETS[args.schemas]:
                result = probe(args.base_url, api_key, model, mode, check)
                print(f"{model} {mode} {check.name}: {result}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe which structured-output modes an OpenAI-compatible endpoint honours, per model.

Usage:
    NVIDIA_API_KEY=... python scripts/probe_schema.py --model <id> [--model <id> ...]

Prints one line per (model, mode): PASS, or FAIL with a reason. PASS means HTTP 200 and a
reply that validates against the schema (needs the ``jsonschema`` package).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, cast, get_args

from voyager_ai.config import NIM_BASE_URL
from voyager_ai.llm import LLMError, OpenAICompatibleClient, SchemaMode

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
    "required": ["city", "days"],
    "additionalProperties": False,
}
MODES: tuple[str, ...] = get_args(SchemaMode)


def validate(text: str) -> str | None:
    """Return None when ``text`` is JSON matching SCHEMA, else the reason it is not."""

    import jsonschema  # imported lazily so importing this script never needs it

    try:
        jsonschema.validate(json.loads(text), SCHEMA)
    except json.JSONDecodeError as exc:
        return f"not JSON ({exc.msg})"
    except jsonschema.ValidationError as exc:
        return f"schema violation ({exc.message})"
    return None


def probe(base_url: str, api_key: str, model: str, mode: str) -> str:
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
            user="Name one city for a short trip and a number of days.",
            json_schema=SCHEMA,
        )
    except LLMError as exc:
        return f"FAIL {exc}"
    problem = validate(response.text)
    return "PASS" if problem is None else f"FAIL {problem}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-url", default=NIM_BASE_URL)
    parser.add_argument("--model", action="append", default=[], required=True)
    parser.add_argument("--mode", action="append", choices=MODES, default=None)
    args = parser.parse_args(argv)

    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_COMPATIBLE_API_KEY", "")
    if not api_key:
        print("Set NVIDIA_API_KEY (or OPENAI_COMPATIBLE_API_KEY).", file=sys.stderr)
        return 2
    for model in args.model:
        for mode in args.mode or MODES:
            print(f"{model} {mode}: {probe(args.base_url, api_key, model, mode)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

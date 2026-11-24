"""Parsers for LLM JSON."""

from __future__ import annotations

import json
import re
from typing import Any

from voyager_ai.models import ValidationError


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks. An unclosed <think> means the answer never arrived."""

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        return ""
    return text.strip()


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
        raise ValidationError(f"The model did not return valid JSON: {exc}") from exc

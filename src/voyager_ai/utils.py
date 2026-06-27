"""Small utility helpers used across VoyagerAI."""

from __future__ import annotations

import calendar
from datetime import date


def clean_text(value: object) -> str:
    """Collapse whitespace while preserving user-provided text."""

    return " ".join(str(value or "").strip().split())


def add_months(start: date, months: int) -> date:
    """Add months without overflowing the target month."""

    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(start.day, last_day))


def format_date(value: date) -> str:
    """Format dates like the original app: 17 May 2026."""

    return value.strftime("%d %b %Y").lstrip("0")


def date_range(months: int, today: date | None = None) -> str:
    """Return a human-readable travel planning window."""

    start = today or date.today()
    end = add_months(start, months)
    return "%s - %s" % (format_date(start), format_date(end))


def truncate(text: str, limit: int = 1200) -> str:
    """Keep prompts bounded while preserving complete-looking text."""

    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."

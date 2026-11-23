"""Retry with exponential backoff for transient LLM provider failures."""

from __future__ import annotations

import random as _random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TypeVar

T = TypeVar("T")


class RetryableError(Exception):
    """A transient failure worth retrying.

    ``retry_after`` is the server's hint in seconds. ``min_delay`` is the shortest wait before
    the next attempt, for failures such as a per-minute quota that a short wait cannot clear.
    """

    def __init__(self, message: str, retry_after: float | None = None, min_delay: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after
        self.min_delay = min_delay


def retry_call(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
    random: Callable[[], float] = _random.random,
) -> T:
    """Call ``fn``, retrying only on RetryableError. Re-raises the last one when exhausted."""

    attempts = max(1, attempts)
    for n in range(1, attempts + 1):
        try:
            return fn()
        except RetryableError as exc:
            if n >= attempts:
                raise
            if exc.retry_after is not None:
                delay = min(max_delay, exc.retry_after)
            else:
                delay = min(max_delay, base_delay * 2 ** (n - 1)) * (1 + 0.25 * random())
            sleep(max(exc.min_delay, delay))
    raise AssertionError("unreachable")  # pragma: no cover


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Parse a Retry-After header (seconds or HTTP date). Returns None if unusable."""

    if value is None:
        return None
    text = value.strip()
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    return max(0.0, (when - current).total_seconds())
